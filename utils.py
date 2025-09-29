# utils.py
import random, json, math, re, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_with_warmup(epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs + 1))
    t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * t))

def class_stratified_split(samples, num_classes, ratios, seed=1234):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for idx, (_, y) in enumerate(samples):
        by_cls[y].append(idx)
    r_tr, r_va, r_at = ratios
    splits = {"train": [], "val": [], "attack": []}
    for c in range(num_classes):
        idxs = by_cls[c]
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * r_tr))
        n_va = int(round(n * r_va))
        n_at = max(0, n - n_tr - n_va)
        splits["train"].extend(idxs[:n_tr])
        splits["val"].extend(idxs[n_tr:n_tr+n_va])
        splits["attack"].extend(idxs[n_tr+n_va:n_tr+n_va+n_at])
    for k in splits:
        rng.shuffle(splits[k])
    return splits

def save_json(path: Path, obj: dict):
    with open(path, "w") as f: json.dump(obj, f)

def load_json(path: Path):
    with open(path, "r") as f: return json.load(f)

def extract_numeric_id(name: str):
    stem = os.path.splitext(name)[0]
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else None

@torch.no_grad()
def norm_tensors_on_device(mean, std, device):
    m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=device).view(1, -1, 1, 1)
    return m, s
