# data.py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode  # NEW

import timm

import config as cfg


from utils import class_stratified_split, load_json, save_json

def build_transforms(input_size, mean, std, target_res=None):
    h, w = input_size[1], input_size[2]
    if target_res is not None:
        h = w = target_res

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            (h, w),
            scale=(0.5, 1.0),         # was (0.6, 1.0)
            ratio=(0.75, 1.6),        # was (0.7, 1.5)
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.25),  # NEW: gentle geom jitter
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(round(h / 0.9)), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def resolve_model_cfg(model_name: str, num_classes_hint=20):
    temp = timm.create_model(model_name, pretrained=True, num_classes=num_classes_hint)
    cfg = timm.data.resolve_model_data_config(temp)
    del temp
    return cfg["mean"], cfg["std"], cfg["input_size"]

def make_loaders(
    data_root: Path, split_file: Path, ratios, batch_size, num_workers, seed, model_name
):
    mean, std, input_size = resolve_model_cfg(model_name)

    # NEW: pull target res if set at runtime (None if not set)
    target_res = getattr(cfg, "TARGET_RES", None)

    # pass it into the transforms
    train_tf, val_tf = build_transforms(input_size, mean, std, target_res=target_res)

    base = datasets.ImageFolder(str(data_root))
    class_names = base.classes
    num_classes = len(class_names)

    if split_file.exists():
        idx = load_json(split_file)
        train_idx, val_idx, attack_idx = idx["train"], idx["val"], idx["attack"]
        print("Loaded saved split:", split_file)
    else:
        idx = class_stratified_split(base.samples, num_classes, ratios, seed=seed)
        train_idx, val_idx, attack_idx = idx["train"], idx["val"], idx["attack"]
        save_json(split_file, idx)
        print("Saved split indices →", split_file.resolve())

    train_ds = Subset(datasets.ImageFolder(str(data_root), transform=train_tf), train_idx)
    val_ds   = Subset(datasets.ImageFolder(str(data_root), transform=val_tf),   val_idx)
    att_ds   = Subset(datasets.ImageFolder(str(data_root), transform=val_tf),   attack_idx)

    train_labels = [base.samples[i][1] for i in train_idx]
    counts = np.bincount(train_labels, minlength=num_classes)
    pct = counts / counts.sum()
    print("Train class %:", (pct * 100).round(2))
    if (pct < 0.03).any():
        print("Using WeightedRandomSampler (class <3%).")
        sample_weights = [1.0 / counts[y] for y in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=bool(num_workers > 0)  # NEW (safe on Win if num_workers>0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0)  # NEW
    )
    att_loader = DataLoader(
        att_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0)  # NEW
    )
    
    if target_res is not None:
        eff_h = eff_w = target_res
    else:
        eff_h, eff_w = input_size[1], input_size[2]
    effective_input_size = (input_size[0], eff_h, eff_w)

    return train_loader, val_loader, att_loader, class_names, effective_input_size, mean, std

# --- inference dataset + helpers ---
class TestImages(torch.utils.data.Dataset):
    def __init__(self, root: Path):
        self.paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            self.paths.extend(sorted(Path(root).glob(ext)))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return img, p.name

def tta_crops_tensor(pil_img, mean_t, std_t, hw, device):
    from torchvision import transforms
    from torchvision.transforms import functional as TF
    H, W = hw
    resize_h = int(round(H / 0.9))
    resize_w = int(round(W / 0.9))
    img_big = TF.resize(pil_img, [resize_h, resize_w])
    crops = transforms.TenCrop((H, W))(img_big)
    tens = [TF.to_tensor(c) for c in crops]
    batch = torch.stack(tens, dim=0).to(device)
    batch = (batch - mean_t) / std_t
    return batch

def center_crop_tensor(pil_img, mean_t, std_t, hw, device):
    from torchvision.transforms import functional as TF
    H, W = hw
    resize_h = int(round(H / 0.9))
    resize_w = int(round(W / 0.9))
    img_big = TF.resize(pil_img, [resize_h, resize_w])
    cc = TF.center_crop(img_big, [H, W])
    x = TF.to_tensor(cc).unsqueeze(0).to(device)
    x = (x - mean_t) / std_t
    return x
