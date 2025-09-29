# model_utils.py
import timm
import torch.nn as nn

def create_model(name: str, num_classes: int, drop_rate: float, drop_path_rate: float):
    return timm.create_model(
        name, pretrained=True, num_classes=num_classes,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate
    )

def freeze_backbone_params(model: nn.Module):
    for n, p in model.named_parameters():
        if not any(k in n.lower() for k in ["head", "classifier", "fc"]):
            p.requires_grad = False

def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True

# -----------------------------
# Differential-LR param groups
# -----------------------------
def get_param_groups_backbone_head(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.1,
    head_lr_mult: float = 1.0,
    no_decay_keywords=("bias", "bn", "norm")
):
    """
    Build optimizer param groups with different LR for backbone vs head,
    and zero weight_decay for norm/bias params.
    Returns a list of dicts suitable for torch.optim (AdamW / SGD).
    """
    bb_decay, bb_nodecay, head_decay, head_nodecay = [], [], [], []
    head_keys = ("head", "classifier", "fc")

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # decide which bucket (head or backbone)
        is_head = any(k in n.lower() for k in head_keys)
        # decide decay or not
        is_no_decay = any(k in n.lower() for k in no_decay_keywords)

        if is_head and not is_no_decay:
            head_decay.append(p)
        elif is_head and is_no_decay:
            head_nodecay.append(p)
        elif not is_head and not is_no_decay:
            bb_decay.append(p)
        else:  # backbone & no_decay
            bb_nodecay.append(p)

    groups = []
    if bb_decay:
        groups.append({"params": bb_decay, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay})
    if bb_nodecay:
        groups.append({"params": bb_nodecay, "lr": base_lr * backbone_lr_mult, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr * head_lr_mult, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": base_lr * head_lr_mult, "weight_decay": 0.0})

    return groups


# -----------------------------
# (Optional) Layer-wise LR decay
# -----------------------------
def get_param_groups_llrd(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.75,
    no_decay_keywords=("bias", "bn", "norm")
):
    """
    Layer-wise LR decay (LLRD): deeper layers get lower LR by a factor (layer_decay^depth).
    Uses timm's helper if available; otherwise falls back to a simple backbone/head split.
    """
    try:
        # timm >= 0.9 provides a nice utility for LLRD:
        from timm.optim.optim_factory import param_groups_layer_decay
        return param_groups_layer_decay(
            model,
            weight_decay=weight_decay,
            lr=base_lr,
            layer_decay=layer_decay,
            no_decay_list=list(no_decay_keywords),
        )
    except Exception:
        # Fallback to simpler backbone/head grouping if helper not available
        return get_param_groups_backbone_head(
            model, base_lr, weight_decay,
            backbone_lr_mult=layer_decay,  # rough approximation
            head_lr_mult=1.0,
            no_decay_keywords=no_decay_keywords
        )