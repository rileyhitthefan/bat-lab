"""Temperature scaling for confidence calibration."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def calibrate_temperature(model, loader, device: str, max_iters=200, lr=5e-2) -> float:
    model.eval()
    temp = model.temperature
    opt = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iters)
    def _nll():
        losses = []
        for x_full, x_call, y, loc, num_feats, _ in loader:
            logits = model(x_full.to(device), x_call.to(device), loc.to(device), num_feats.to(device))
            losses.append(F.cross_entropy(logits, y.to(device)))
        return torch.stack(losses).mean()
    def closure():
        opt.zero_grad()
        loss = _nll()
        loss.backward()
        return loss
    opt.step(closure)
    return float(temp.item())
