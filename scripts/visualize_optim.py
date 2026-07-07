#!/usr/bin/env python3
"""Graphes pour comprendre l'etape Criterion / Optim.

Produit 2 figures dans figures/ :
  1. overfit_batch.png  -- loss & accuracy sur un VRAI batch sur-appris :
     preuve que criterion + optimizer descendent bien la loss.
  2. lr_range_test.png  -- LR range test (loss vs learning rate) :
     justifie le choix lr=1e-3 en montrant la zone de descente saine.

Usage:
    python scripts/visualize_optim.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend fichier
import matplotlib.pyplot as plt

import torch

from dataloader import make_loaders
from model import MovementCNN
from train_setup import build_training

FIG_DIR = Path(r"C:/Users/pierr/dev/iot/figures")
LR_CHOISI = 1e-3


# --------------------------------------------------------------------------
# 1. Sur-apprentissage d'un vrai batch : preuve de descente
# --------------------------------------------------------------------------
def plot_overfit_batch(xb, yb, n_steps=200, seed=0):
    """Sur-apprend un unique batch reel avec le trio de build_training.
    La loss doit s'effondrer, l'accuracy monter a 1.0 : la machinerie apprend."""
    torch.manual_seed(seed)
    model = MovementCNN()
    criterion, optimizer, _ = build_training(model)

    losses, accs = [], []
    model.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append((out.argmax(1) == yb).float().mean().item())

    fig, ax1 = plt.subplots(figsize=(9, 5))
    c_loss, c_acc = "#d62728", "#2ca02c"
    ax1.plot(losses, color=c_loss, linewidth=1.8, label="loss")
    ax1.set_xlabel("etape d'optimisation")
    ax1.set_ylabel("loss (CrossEntropy)", color=c_loss)
    ax1.tick_params(axis="y", labelcolor=c_loss)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    ax2 = ax1.twinx()
    ax2.plot(accs, color=c_acc, linewidth=1.8, label="accuracy")
    ax2.set_ylabel("accuracy", color=c_acc)
    ax2.tick_params(axis="y", labelcolor=c_acc)
    ax2.set_ylim(-0.02, 1.05)

    ax1.set_title(
        f"Sur-apprentissage d'un batch reel ({len(yb)} ech.)  "
        f"loss {losses[0]:.2f} -> {losses[-1]:.3f}\n"
        "preuve : criterion + Adam font descendre la loss")
    fig.tight_layout()
    out = FIG_DIR / "overfit_batch.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 2. LR range test : loss vs learning rate
# --------------------------------------------------------------------------
def plot_lr_range_test(train_loader, lr_min=1e-6, lr_max=1.0,
                       n_steps=120, seed=0):
    """Augmente le lr exponentiellement de lr_min a lr_max sur n_steps,
    en consommant des batches reels ; trace la loss (lissee) vs lr.
    La bonne zone = la pente descendante la plus raide, avant l'explosion."""
    torch.manual_seed(seed)
    model = MovementCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_min, weight_decay=1e-4)

    gamma = (lr_max / lr_min) ** (1.0 / (n_steps - 1))
    lrs, raw = [], []

    model.train()
    it = iter(train_loader)
    for step in range(n_steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader)
            xb, yb = next(it)

        lr = lr_min * (gamma ** step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        raw.append(loss.item())
        if not torch.isfinite(loss):  # loss explosee : inutile d'aller plus loin
            break

    # Lissage exponentiel pour lire la tendance malgre le bruit des batches.
    smooth, beta, avg = [], 0.8, raw[0]
    for v in raw:
        avg = beta * avg + (1 - beta) * v
        smooth.append(avg)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lrs, raw, color="#c7c7c7", linewidth=1.0, label="loss brute")
    ax.plot(lrs, smooth, color="#1f77b4", linewidth=2.2, label="loss lissee")
    ax.axvline(LR_CHOISI, color="#d62728", linewidth=1.6, linestyle="--",
               label=f"lr choisi = {LR_CHOISI:g}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (echelle log)")
    ax.set_ylabel("loss (CrossEntropy)")
    ax.set_title("LR range test  --  loss vs learning rate\n"
                 "zone saine = descente la plus raide avant l'explosion")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "lr_range_test.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Chargement d'un batch reel...")
    train_loader, _, _ = make_loaders(batch_size=64)
    xb, yb = next(iter(train_loader))

    outs = [
        plot_overfit_batch(xb, yb),
        plot_lr_range_test(train_loader),
    ]
    print("\nFigures ecrites :")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
