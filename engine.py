import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks import pgd_attack
from augment import apply_cutmix, apply_mixup

class Trainer:
    def __init__(self, cfg, device, mean_t, std_t):
        self.cfg = cfg
        self.device = device
        self.mean_t = mean_t
        self.std_t = std_t

    def _attack_params_for_epoch(self, epoch: int):
        """
        Returns (steps, alpha, random_start) for the given epoch (0-based).
        Curriculum respects WARMUP_FGSM_EPOCHS and ramps steps/alpha.
        """
        cfg = self.cfg
        e = epoch + 1  # 1-based

        if not getattr(cfg, "CURRICULUM", False):
            if e <= cfg.WARMUP_FGSM_EPOCHS:
                return 1, cfg.EPS, False   # FGSM: alpha = eps, no random start
            else:
                return cfg.PGD_STEPS, cfg.ALPHA, True

        # ---- Curriculum enabled ----
        warm = cfg.WARMUP_FGSM_EPOCHS

        # Phase 1: FGSM warmup with gentler alpha
        if e <= warm:
            alpha_fgsm = cfg.EPS * (getattr(cfg, "CURR_ALPHA_FRAC", 1/4) if getattr(cfg, "CURR_USE_ALPHA_FRACTION", False) else 1.0)
            return 1, alpha_fgsm, False

        # Phase 2+: PGD ramp after warmup
        after = e - warm
        if after <= 3:
            steps = 5
        elif after <= 8:
            steps = 10
        else:
            steps = min(cfg.PGD_STEPS, 20)

        # alpha ramp: from EPS*frac to ALPHA over ~10 epochs after warmup
        if getattr(cfg, "CURR_USE_ALPHA_FRACTION", False):
            t = min(1.0, after / 10.0)
            alpha_start = cfg.EPS * getattr(cfg, "CURR_ALPHA_FRAC", 1/4)
            alpha = alpha_start + t * (cfg.ALPHA - alpha_start)
        else:
            alpha = cfg.ALPHA

        return steps, alpha, True

    def train_one_epoch(self, model, loader, optimizer, scaler, epoch):
        cfg = self.cfg
        model.train()
        running = 0.0

        # Decide attack schedule ONCE per epoch
        steps, step_alpha, rnd = self._attack_params_for_epoch(epoch)
        print(f"[curriculum] epoch {epoch+1}: steps={steps}, alpha={step_alpha:.6f}, rnd={rnd}")

        # Ramp adversarial weight from 0 → ADV_MAX_WEIGHT over ADV_RAMP_EPOCHS
        lam_adv = min(cfg.ADV_MAX_WEIGHT, (epoch + 1) / max(1, cfg.ADV_RAMP_EPOCHS) * cfg.ADV_MAX_WEIGHT)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)   # [0,1]
            labels = labels.to(self.device)

            # Craft adversarial batch in full precision (disable autocast inside attack loop)
            with torch.amp.autocast('cuda', enabled=False):
                adv = pgd_attack(
                    model, images, labels,
                    eps=cfg.EPS, alpha=step_alpha, steps=steps,
                    mean_t=self.mean_t, std_t=self.std_t, random_start=rnd
                )

            optimizer.zero_grad(set_to_none=True)

            used_aug = False
            # Probabilistic CutMix to reduce turbulence on small datasets
            if cfg.USE_CUTMIX and (random.random() < getattr(cfg, "CUTMIX_PROB", 1.0)) and images.size(0) >= 4:
                half = images.size(0) // 2
                A_imgs, A_lbls = adv[:half], labels[:half]
                B_imgs, B_lbls = images[half:], labels[half:]
                B_mix, B_lbls2, lam_eff = apply_cutmix(B_imgs, B_lbls, cfg.CUTMIX_ALPHA, self.device)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits_adv = model((A_imgs - self.mean_t) / self.std_t)
                    loss_adv = F.cross_entropy(logits_adv, A_lbls, label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0))

                    logits_cut = model((B_mix - self.mean_t) / self.std_t)
                    loss_cut = (lam_eff * F.cross_entropy(logits_cut, B_lbls, reduction='none',
                                                          label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0)) +
                                (1. - lam_eff) * F.cross_entropy(logits_cut, B_lbls2, reduction='none',
                                                                 label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0))
                                ).mean()

                    # adv:clean weighting (CutMix acts as "clean" branch here)
                    loss = lam_adv * loss_adv + (1.0 - lam_adv) * loss_cut
                used_aug = True

            elif cfg.USE_MIXUP and images.size(0) >= 4:
                half = images.size(0) // 2
                A_imgs, A_lbls = adv[:half], labels[:half]
                B_imgs, B_lbls = images[half:], labels[half:]
                B_mix, B_lbls2, lam = apply_mixup(B_imgs, B_lbls, cfg.MIXUP_ALPHA, self.device)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits_adv = model((A_imgs - self.mean_t) / self.std_t)
                    loss_adv = F.cross_entropy(logits_adv, A_lbls, label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0))

                    logits_mix = model((B_mix - self.mean_t) / self.std_t)
                    ce1 = F.cross_entropy(logits_mix, B_lbls, reduction='none', label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0)).mean()
                    ce2 = F.cross_entropy(logits_mix, B_lbls2, reduction='none', label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0)).mean()
                    loss_mix = lam * ce1 + (1 - lam) * ce2

                    loss = lam_adv * loss_adv + (1.0 - lam_adv) * loss_mix
                used_aug = True

            if not used_aug:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits_adv = model((adv - self.mean_t) / self.std_t)
                    logits_clean = model((images - self.mean_t) / self.std_t)
                    ce_adv   = F.cross_entropy(logits_adv, labels, label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0))
                    ce_clean = F.cross_entropy(logits_clean, labels, label_smoothing=getattr(cfg, "LABEL_SMOOTH", 0.0))
                    loss = lam_adv * ce_adv + (1.0 - lam_adv) * ce_clean

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

            # Optional: cap batches per epoch to keep runs cool/small
            cap = getattr(cfg, "CAP_STEPS", None)
            if cap is not None and (batch_idx + 1) >= cap:
                break

        denom = min(len(loader), getattr(cfg, "CAP_STEPS", len(loader)) or len(loader))
        return running / max(1, denom)

    @torch.no_grad()
    def evaluate_clean(self, model, loader):
        """Standard validation on clean images."""
        model.eval()
        tot = 0
        correct = 0
        loss_sum = 0.0
        for x, y in loader:
            x = x.to(self.device)     # [0,1]
            y = y.to(self.device)
            logits = model((x - self.mean_t) / self.std_t)
            loss_sum += F.cross_entropy(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        avg_loss = loss_sum / max(1, tot)
        acc = correct / max(1, tot)
        return avg_loss, acc

    def evaluate_pgd(self, model, loader):
        """
        Robust accuracy under PGD with cfg.PGD_STEPS and cfg.ALPHA (pixel-space).
        We keep model in eval() for BN/Dropout, but enable grads to craft adversarials.
        """
        cfg = self.cfg
        model.eval()
        tot = 0
        correct = 0

        for x, y in loader:
            x = x.to(self.device)     # [0,1]
            y = y.to(self.device)

            # grads ON to create adversarial examples
            with torch.enable_grad():
                adv = pgd_attack(
                    model, x, y,
                    eps=cfg.EPS, alpha=cfg.ALPHA, steps=cfg.PGD_STEPS,
                    mean_t=self.mean_t, std_t=self.std_t, random_start=True
                )

            # measure accuracy with grads OFF
            with torch.no_grad():
                logits = model((adv - self.mean_t) / self.std_t)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)

        acc = correct / max(1, tot)
        return acc
