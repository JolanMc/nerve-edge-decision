"""Deterministic CPU training of EdgeNet on PlanesNet.

Augmentation is geometric only (rotations by multiples of 90 degrees and
flips), which nadir imagery is invariant to. There is deliberately no
photometric augmentation, so that the sensor faults used later stay outside
the training distribution.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import EdgeNet


def augment(xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    xb = xb.clone()
    k = torch.randint(0, 4, (xb.shape[0],), generator=gen)
    flip = torch.randint(0, 2, (xb.shape[0],), generator=gen).bool()
    for r in range(1, 4):
        idx = k == r
        if idx.any():
            xb[idx] = torch.rot90(xb[idx], r, dims=(2, 3))
    if flip.any():
        xb[flip] = xb[flip].flip(3)
    return xb


@torch.no_grad()
def predict_logits(net: torch.nn.Module, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(net(torch.from_numpy(x[i : i + batch])).numpy())
    return np.concatenate(out)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch: int = 128,
    lr: float = 3e-3,
    weight_decay: float = 5e-4,
    seed: int = 0,
    threads: int = 2,
    log=print,
):
    """Train EdgeNet; return the network at its best validation accuracy and the history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)

    net = EdgeNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(x_train)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator().manual_seed(seed)
    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)

    best_acc, best_state, best_epoch = -1.0, None, -1
    history = []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        net.train()
        perm = torch.randperm(n, generator=gen)
        tot_loss, tot_ok = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = augment(xt[idx], gen)
            yb = yt[idx]
            logits = net(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_ok += int((logits.argmax(1) == yb).sum())
        val_acc = float((predict_logits(net, x_val).argmax(1) == y_val).mean())
        row = {
            "epoch": ep,
            "train_loss": tot_loss / n,
            "train_acc": tot_ok / n,
            "val_acc": val_acc,
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        log(
            f"epoch {ep:2d}  loss {row['train_loss']:.4f}  train {row['train_acc']:.4f}  "
            f"val {val_acc:.4f}  ({row['seconds']:.0f} s)"
        )
        if val_acc > best_acc:
            best_acc, best_state, best_epoch = val_acc, copy.deepcopy(net.state_dict()), ep
    net.load_state_dict(best_state)
    net.eval()
    return net, {"history": history, "best_epoch": best_epoch, "best_val_acc": best_acc}
