# main.py
import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from utils import set_seed, norm_tensors_on_device, cosine_with_warmup
from data import make_loaders
from model_utils import (
    create_model, freeze_backbone_params, unfreeze_all,
    get_param_groups_backbone_head, get_param_groups_llrd  # noqa: F401 (llrd available if you want it)
)
from engine import Trainer
from inference import submit_predictions
from checkpoints import save_state_dict, save_checkpoint, make_ckpt, load_checkpoint


# ---------- DEBUG ----------
def _print_group_lrs(optimizer, tag):
    print(f"[LR groups: {tag}]")
    for i, g in enumerate(optimizer.param_groups):
        print(f"  group {i}: lr={g['lr']:.3e}, weight_decay={g.get('weight_decay', None)}")


def log_current_lrs(optimizer, epoch):
    lrs = [g["lr"] for g in optimizer.param_groups]
    lrs_str = " | ".join([f"group{i}: {lr:.3e}" for i, lr in enumerate(lrs)])
    print(f"[Epoch {epoch+1}] LRs → {lrs_str}")


def tb_log_lrs(writer, optimizer, epoch):
    for i, g in enumerate(optimizer.param_groups):
        writer.add_scalar(f"lr/group_{i}", g["lr"], epoch)


def tb_log_metrics(writer, tr_loss, va_loss, va_acc, rob_acc, epoch):
    writer.add_scalar("metrics/train_loss", tr_loss, epoch)
    writer.add_scalar("metrics/val_loss", va_loss, epoch)
    writer.add_scalar("metrics/val_acc", va_acc, epoch)
    writer.add_scalar("metrics/rob_acc", rob_acc, epoch)


def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss"); plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x*100 for x in history["val_acc"]], label="Val Acc (%)")
    plt.plot(epochs, [x*100 for x in history["rob_acc"]], label="Robust Acc (%)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation & Robust Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    # plt.show()  # optional


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="", help="Path to .ckpt/.pth OR directory containing ckpts")
    ap.add_argument("--resume-strict", action="store_true", help="Strict model state load")
    ap.add_argument("--run_name", type=str, default="exp1", help="TensorBoard/ckpt subdir")

    # --- fast / cool knobs ---
    ap.add_argument("--num_epochs", type=int, default=None, help="Override cfg.NUM_EPOCHS for a short run")
    ap.add_argument("--cap_steps", type=int, default=None, help="Max batches per epoch (e.g., 100)")
    ap.add_argument("--fgsm_only", action="store_true", help="Use FGSM (1-step) instead of multi-step PGD")
    ap.add_argument("--pgd_steps", type=int, default=None, help="Override cfg.PGD_STEPS (e.g., 5)")
    ap.add_argument("--keep_frozen", action="store_true", help="Keep backbone frozen (head-only)")
    ap.add_argument("--target_res", type=int, default=None, help="Resize train/val to this square res (e.g., 192)")
    return ap.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name

    # Per-run checkpoint directory
    ckpt_dir = (cfg.CHECKPOINT_DIR / run_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- runtime overrides (fast mode) ---
    if args.num_epochs is not None:
        cfg.NUM_EPOCHS = args.num_epochs
    if args.fgsm_only:
        cfg.WARMUP_FGSM_EPOCHS = cfg.NUM_EPOCHS  # FGSM entire run
        cfg.PGD_STEPS = 1
    if args.pgd_steps is not None:
        cfg.PGD_STEPS = args.pgd_steps
    if args.keep_frozen:
        cfg.FREEZE_EPOCHS = 10**9               # never unfreeze
    if getattr(args, "target_res", None) is not None:
        setattr(cfg, "TARGET_RES", args.target_res)
    if getattr(args, "cap_steps", None) is not None:
        setattr(cfg, "CAP_STEPS", args.cap_steps)

    # Setup
    set_seed(cfg.SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # TB writer
    writer = SummaryWriter(log_dir=str(cfg.PROJECT_DIR / "tb_runs" / run_name))

    # Data (TARGET_RES is applied inside make_loaders → build_transforms)
    train_loader, val_loader, att_loader, class_names, input_size, mean, std = make_loaders(
        cfg.DATA_ROOT, cfg.SPLIT_FILE, cfg.SPLIT_RATIOS, cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.SEED, cfg.MODEL_NAME
    )
    print(f"Input size: {input_size} | mean: {mean} | std: {std}")
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Attack-Val: {len(att_loader)}")

    # Model
    model = create_model(
        cfg.MODEL_NAME, num_classes=len(class_names),
        drop_rate=cfg.DROP_RATE, drop_path_rate=cfg.DROP_PATH
    ).to(device)

    # ----- Resume & param groups -----
    start_epoch = 0
    resume_ckpt = None
    did_unfreeze = False  # NEW: track whether we've already unfreezed

    if args.resume:
        resume_ckpt = Path(args.resume)
        if resume_ckpt.is_dir():
            candidates = sorted([p for p in resume_ckpt.glob("*.ckpt")],
                                key=lambda p: p.stat().st_mtime)
            assert candidates, f"No .ckpt files in {resume_ckpt}"
            resume_ckpt = candidates[-1]

        # Peek "epoch" (if .pth, will likely be -1). This only decides the initial grouping.
        tmp = torch.load(resume_ckpt, map_location=device)
        last_epoch = int(tmp.get("epoch", -1)) if isinstance(tmp, dict) else -1
        start_epoch = last_epoch + 1
        print(f"[resume] Found checkpoint at epoch {last_epoch} → start_epoch {start_epoch}")

        # Decide initial param groups before the loop:
        if start_epoch >= cfg.FREEZE_EPOCHS:
            # Start with full fine-tuning groups
            unfreeze_all(model)
            param_groups = get_param_groups_backbone_head(
                model, base_lr=cfg.LR * 0.5, weight_decay=cfg.WEIGHT_DECAY,
                backbone_lr_mult=0.1, head_lr_mult=1.0
            )
            optimizer = torch.optim.AdamW(param_groups)
            # remaining horizon from current start
            remaining = max(1, cfg.NUM_EPOCHS - start_epoch)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda e: cosine_with_warmup(e, remaining, 0))
            _print_group_lrs(optimizer, "resume: full fine-tune")
            did_unfreeze = True
        else:
            # Resume but still in head-only warmup
            freeze_backbone_params(model)
            param_groups = get_param_groups_backbone_head(
                model, base_lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                backbone_lr_mult=0.0, head_lr_mult=1.0
            )
            optimizer = torch.optim.AdamW(param_groups)
            warmup_epochs = max(1, cfg.NUM_EPOCHS // 20)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda e: cosine_with_warmup(e, cfg.NUM_EPOCHS, warmup_epochs))
            _print_group_lrs(optimizer, "resume: head-only warmup")
    else:
        # Fresh run → head-only warmup groups
        freeze_backbone_params(model)
        param_groups = get_param_groups_backbone_head(
            model, base_lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
            backbone_lr_mult=0.0, head_lr_mult=1.0
        )
        optimizer = torch.optim.AdamW(param_groups)
        warmup_epochs = max(1, cfg.NUM_EPOCHS // 20)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: cosine_with_warmup(e, cfg.NUM_EPOCHS, warmup_epochs))
        _print_group_lrs(optimizer, "head-only warmup")

    # Scaler & normalization tensors
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    mean_t, std_t = norm_tensors_on_device(mean, std, device)
    trainer = Trainer(cfg, device, mean_t, std_t)

    # Load states (model/optim/sched/scaler) if resuming; carry over best metrics if available
    if resume_ckpt is not None:
        state = load_checkpoint(
            resume_ckpt, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            device=device, strict=args.resume_strict
        )
        start_epoch = state.start_epoch
        best_rob = float(state.metrics.get("robust_acc", -1.0))
        best_val = float(state.metrics.get("val_acc", -1.0))
        best_epoch_rob, best_epoch_val = -1, -1  # optional: restore if saved in meta
    else:
        best_rob, best_val = -1.0, -1.0
        best_epoch_rob, best_epoch_val = -1, -1

    bad_epochs = 0
    meta = {
        "MODEL_NAME": cfg.MODEL_NAME,
        "LR": cfg.LR, "WEIGHT_DECAY": cfg.WEIGHT_DECAY, "DROP_RATE": cfg.DROP_RATE, "DROP_PATH": cfg.DROP_PATH,
        "EPS": cfg.EPS, "ALPHA": cfg.ALPHA, "PGD_STEPS": cfg.PGD_STEPS, "WARMUP_FGSM_EPOCHS": cfg.WARMUP_FGSM_EPOCHS,
        "USE_TTA": cfg.USE_TTA, "USE_MIXUP": cfg.USE_MIXUP, "MIXUP_ALPHA": cfg.MIXUP_ALPHA,
        "USE_CUTMIX": cfg.USE_CUTMIX, "CUTMIX_ALPHA": cfg.CUTMIX_ALPHA,
        "SPLIT_FILE": str(cfg.SPLIT_FILE), "DATA_ROOT": str(cfg.DATA_ROOT),
        "INPUT_SIZE": input_size, "MEAN": mean, "STD": std,
    }
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "rob_acc": []}

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):

        # Unfreeze inside loop only if we haven't already
        if (not did_unfreeze) and (epoch == cfg.FREEZE_EPOCHS):
            print("Unfreezing backbone for full fine-tuning…")
            unfreeze_all(model)
            param_groups = get_param_groups_backbone_head(
                model, base_lr=cfg.LR * 0.5, weight_decay=cfg.WEIGHT_DECAY,
                backbone_lr_mult=0.1, head_lr_mult=1.0
            )
            optimizer = torch.optim.AdamW(param_groups)
            _print_group_lrs(optimizer, "full fine-tune")
            # rebase scheduler on the remaining horizon from *this* epoch
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda e: cosine_with_warmup(e, cfg.NUM_EPOCHS - epoch, 0)
            )
            did_unfreeze = True

        tr_loss = trainer.train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        va_loss, va_acc = trainer.evaluate_clean(model, val_loader)
        rob_acc = trainer.evaluate_pgd(model, att_loader)

        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | train loss {tr_loss:.4f} | "
              f"val {va_loss:.4f}/{va_acc*100:.2f}% | attack-val robust {rob_acc*100:.2f}%")

        scheduler.step()
        log_current_lrs(optimizer, epoch)
        tb_log_lrs(writer, optimizer, epoch)
        tb_log_metrics(writer, tr_loss, va_loss, va_acc, rob_acc, epoch)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["rob_acc"].append(rob_acc)

        # Save-on-improvement: robust
        if rob_acc > best_rob:
            best_rob, best_epoch_rob = rob_acc, epoch
            bad_epochs = 0
            # save inside run folder
            save_state_dict(ckpt_dir / "best_robust.pth", model)
            ckpt = make_ckpt(
                model, optimizer, scheduler, scaler, epoch, tag="best_robust",
                metrics={"robust_acc": best_rob, "val_acc": va_acc, "val_loss": va_loss}, meta=meta
            )
            save_checkpoint(ckpt_dir / "best_robust.ckpt", ckpt)
            # convenience alias
            save_state_dict(cfg.PROJECT_DIR / "timm_adv_best.pth", model)
            print(f"  ✓ New best robust acc = {best_rob*100:.2f}% → saved checkpoints.")
        else:
            bad_epochs += 1

        # Save-on-improvement: validation
        if va_acc > best_val:
            best_val, best_epoch_val = va_acc, epoch
            save_state_dict(ckpt_dir / "best_val.pth", model)
            ckpt = make_ckpt(
                model, optimizer, scheduler, scaler, epoch, tag="best_val",
                metrics={"robust_acc": rob_acc, "val_acc": best_val, "val_loss": va_loss}, meta=meta
            )
            save_checkpoint(ckpt_dir / "best_val.ckpt", ckpt)
            print(f"  ✓ New best val acc = {best_val*100:.2f}% → saved checkpoints.")

                # --- Always save "last" checkpoint for exact resume (even if no improvement) ---
        ckpt_last = make_ckpt(
            model, optimizer, scheduler, scaler, epoch, tag="last",
            metrics={"robust_acc": rob_acc, "val_acc": va_acc, "val_loss": va_loss}, meta=meta
        )
        save_checkpoint(ckpt_dir / "last.ckpt", ckpt_last)
        # (optional) also keep a plain state_dict for quick weight-only loads:
        # save_state_dict(ckpt_dir / "last.pth", model)

        if bad_epochs >= cfg.PATIENCE:
            print("Early stopping on robust metric.")
            break
        
    plot_training_curves(history)

    # Load best robust for final reporting / inference
    best_path = ckpt_dir / "best_robust.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best robust epoch {best_epoch_rob+1} (robust acc {best_rob*100:.2f}%).")

    va_loss, va_acc = trainer.evaluate_clean(model, val_loader)
    rob_acc = trainer.evaluate_pgd(model, att_loader)
    print(f"Final Clean Val Acc: {va_acc*100:.2f}% | Final Robust (PGD) Acc: {rob_acc*100:.2f}%")

    submit_predictions(model, class_names, mean_t, std_t, input_size, cfg.TEST_DIR, cfg.USE_TTA, device)
    writer.close()


if __name__ == "__main__":
    main()


# Examples:
# python main.py --run_name exp10 --num_epochs 10
# python main.py --resume checkpoints/exp10 --run_name exp10_resume
# python main.py --resume checkpoints/exp10/best_robust.ckpt --run_name exp10_resume
# python main.py --resume checkpoints/exp10/best_robust.pth --run_name exp10_restart
# python main.py --resume kaggle_model/checkpoints/exp1_4_restart40/best_robust.ckpt                --run_name exp1_4_breadth_pgd5              --num_epochs 50
# python main.py --resume kaggle_model/checkpoints/exp1_4_breadth_pgd5/best_robust.ckpt                --run_name exp1_5_PGD10              --num_epochs 37