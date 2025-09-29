# checkpoints.py
import torch
from pathlib import Path
from typing import Dict, Any, Optional, NamedTuple

class ResumeState(NamedTuple):
    start_epoch: int
    last_epoch: int
    tag: Optional[str]
    metrics: Dict[str, Any]
    meta: Dict[str, Any]

def save_state_dict(path_pth: Path, model):
    torch.save(model.state_dict(), path_pth)

def save_checkpoint(path_ckpt: Path, state: Dict[str, Any]):
    torch.save(state, path_ckpt)

def make_ckpt(model, optimizer, scheduler, scaler, epoch, tag, metrics: dict, meta: dict):
    return {
        "tag": tag,
        "epoch": int(epoch),                 # last finished epoch (0-indexed)
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics or {},
        "meta": meta or {},
    }

def load_checkpoint(
    path_ckpt: Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> ResumeState:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path_ckpt, map_location=device)

    # 1) model
    load_res = model.load_state_dict(ckpt.get("state_dict", {}), strict=strict)
    if isinstance(load_res, tuple):
        missing, unexpected = load_res
        if missing or unexpected:
            print(f"[load_checkpoint] Warning missing={missing}, unexpected={unexpected}")

    # 2) opt/sched/scaler
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[load_checkpoint] AMP scaler load failed (non-fatal): {e}")

    last_epoch = int(ckpt.get("epoch", -1))
    start_epoch = last_epoch + 1
    tag   = ckpt.get("tag", None)
    metrics = ckpt.get("metrics", {})
    meta    = ckpt.get("meta", {})

    print(f"[load_checkpoint] Loaded '{path_ckpt}' (tag={tag}) at epoch {last_epoch}. Resuming from epoch {start_epoch}.")
    return ResumeState(start_epoch=start_epoch, last_epoch=last_epoch, tag=tag, metrics=metrics, meta=meta)
