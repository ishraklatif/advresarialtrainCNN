# augment.py
import math, random
import numpy as np
import torch
import torch.nn.functional as F

def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, max(0, W - 1))
    cy = random.randint(0, max(0, H - 1))
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def apply_cutmix(clean_imgs, clean_lbls, alpha, device):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(clean_imgs.size(0), device=device)
    imgs2, lbls2 = clean_imgs[perm], clean_lbls[perm]
    _, _, H, W = clean_imgs.shape
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    mixed = clean_imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs2[:, :, y1:y2, x1:x2]
    lam_eff = 1. - ((x2 - x1) * (y2 - y1) / (W * H))
    return mixed, lbls2, lam_eff

def apply_mixup(clean_imgs, clean_lbls, alpha, device):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(clean_imgs.size(0), device=device)
    imgs2, lbls2 = clean_imgs[perm], clean_lbls[perm]
    mixed = lam * clean_imgs + (1 - lam) * imgs2
    return mixed, lbls2, lam
