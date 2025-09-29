# inference.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from utils import extract_numeric_id
from data import TestImages, tta_crops_tensor, center_crop_tensor

def submit_predictions(model, class_names, mean_t, std_t, input_size, test_dir: Path, use_tta: bool, device):
    if not test_dir.exists():
        print("Test dir not found:", test_dir.resolve())
        return
    test_ds = TestImages(test_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=lambda b: b)  # identity collate

    rows = []
    skipped = 0
    H, W = input_size[1], input_size[2]

    with torch.no_grad():
        for batch in test_loader:
            pil_img, name = batch[0]

            if use_tta:
                xb = tta_crops_tensor(pil_img, mean_t, std_t, hw=(H, W), device=device)
            else:
                xb = center_crop_tensor(pil_img, mean_t, std_t, hw=(H, W), device=device)

            logits = model(xb)                # [TTA, K]
            logits = logits.mean(dim=0, keepdim=True)
            pred_idx = logits.argmax(1).item()
            pred_label = class_names[pred_idx]

            num_id = extract_numeric_id(name)
            if num_id is None:
                skipped += 1
                continue
            rows.append((num_id, pred_label))

    if skipped:
        print(f"[submission] Skipped {skipped} files without numeric IDs in filename.")

    df = pd.DataFrame(rows, columns=["ID", "Label"]).sort_values("ID")
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv  (columns: ID, Label)  | rows:", len(df))
