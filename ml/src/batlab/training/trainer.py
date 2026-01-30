"""
This file contains the main training loop for the model.

It defines functions to:
- run one training or validation epoch
- track loss and accuracy
- apply early stopping
- save the best-performing model checkpoint

Separating this logic from the training script keeps the code modular and easier
to reason about.
"""
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
        x_full = x_full.to(device)
        x_call = x_call.to(device)
        y = y.to(device)
        loc = loc.to(device)
        num_feats = num_feats.to(device)

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

    best_val = 0.0
    history = []
    best_path = os.path.join(model_dir, "best_model.pt")
    patience_counter = 0
    early_stop_patience = 7

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loaders["train"], device, opt)
        va = run_epoch(model, loaders["val"], device, None)
        scheduler.step(va["acc"])
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": opt.param_groups[0]["lr"]})

        if va["acc"] > best_val:
            best_val = va["acc"]
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "meta": meta}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    return TrainResult(best_path=best_path, history=history)
