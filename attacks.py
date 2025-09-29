# attacks.py
import torch
import torch.nn.functional as F

def pgd_attack(model, x_01, y, eps, alpha, steps, mean_t, std_t, random_start=True):
    """
    x_01 in [0,1]. mean_t/std_t: [1,C,1,1].
    """
    x_adv = x_01.clone().detach()
    if random_start:
        x_adv = x_adv + (torch.rand_like(x_adv) * 2 * eps - eps)
        x_adv = x_adv.clamp(0., 1.)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model((x_adv - mean_t) / std_t)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_01, min=-eps, max=eps)
            x_adv = (x_01 + delta).clamp(0., 1.)
    return x_adv.detach()
