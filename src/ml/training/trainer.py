"""Training loop and checkpointing."""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Dict, Any
import torch
from tqdm import tqdm
from ..models.net import accuracy, cross_entropy_loss

@dataclass
class TrainResult:
    best_path: str
    history: list

def run_epoch(model, loader, device: str, opt=None) -> Dict[str, float]:
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for x_full, x_call, y, loc, num_feats, _ in pbar:
        x_full, x_call = x_full.to(device), x_call.to(device)
        y, loc, num_feats = y.to(device), loc.to(device), num_feats.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x_full, x_call, loc, num_feats)
            loss = cross_entropy_loss(logits, y)
            if is_train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        bs = x_full.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        total_n += bs
        pbar.set_postfix(loss=total_loss/total_n, acc=total_acc/total_n)
    return {"loss": total_loss/total_n, "acc": total_acc/total_n}

def train_model(model, loaders: Dict[str, Any], device: str, lr: float, epochs: int, model_dir: str, meta: dict) -> TrainResult:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    best_val, history, patience_counter = 0.0, [], 0
    best_path = os.path.join(model_dir, "best_model.pt")
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loaders["train"], device, opt)
        va = run_epoch(model, loaders["val"], device, None)
        scheduler.step(va["acc"])
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": opt.param_groups[0]["lr"]})
        if va["acc"] > best_val:
            best_val, patience_counter = va["acc"], 0
            torch.save({"model_state": model.state_dict(), "meta": meta}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= 7:
                break
    return TrainResult(best_path=best_path, history=history)
