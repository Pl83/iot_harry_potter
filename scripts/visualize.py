#!/usr/bin/env python3
"""Graphes pour comprendre le dataset et le modele.

Produit 3 figures dans figures/ :
  1. signals_per_class.png  -- signaux ax/ay/az types de chaque classe.
  2. class_distribution.png -- effectifs par classe dans train/val/test.
  3. architecture.png       -- flux des formes de tenseur dans le CNN.

Usage:
    python scripts/visualize.py
"""
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend fichier, pas de fenetre
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

from dataloader import (
    _load_samples, _stratified_split, DATA_CSV, CLASSES, SEED,
)
from model import MovementCNN, N_CHANNELS, SEQ_LEN

FIG_DIR = Path(r"C:/Users/pierr/dev/iot/figures")
CH_NAMES = ["ax", "ay", "az"]
CH_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]


# --------------------------------------------------------------------------
# 1. Signaux types par classe
# --------------------------------------------------------------------------
def plot_signals_per_class(samples, n_examples=5, seed=SEED):
    """Une ligne par classe. Trace n_examples signaux (fins) + la moyenne
    (epaisse) pour chaque canal ax/ay/az."""
    by_class = defaultdict(list)
    for x, y, _ in samples:
        by_class[y].append(x)

    rng = random.Random(seed)
    fig, axes = plt.subplots(len(CLASSES), N_CHANNELS,
                             figsize=(12, 3 * len(CLASSES)),
                             sharex=True, squeeze=False)

    for row, cls_idx in enumerate(sorted(by_class)):
        xs = by_class[cls_idx]
        pick = rng.sample(xs, min(n_examples, len(xs)))
        stack = torch.stack(xs)              # (N, 3, 100)
        mean = stack.mean(dim=0)             # (3, 100)
        for ch in range(N_CHANNELS):
            ax = axes[row][ch]
            for x in pick:
                ax.plot(x[ch].numpy(), color=CH_COLORS[ch],
                        alpha=0.25, linewidth=0.8)
            ax.plot(mean[ch].numpy(), color=CH_COLORS[ch], linewidth=2.2)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            if row == 0:
                ax.set_title(CH_NAMES[ch], fontsize=11)
            if ch == 0:
                ax.set_ylabel(f"{CLASSES[cls_idx]}\n(z-score)", fontsize=10)
            if row == len(CLASSES) - 1:
                ax.set_xlabel("pas de temps")

    fig.suptitle("Signaux types par classe  (fin = exemples, epais = moyenne)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = FIG_DIR / "signals_per_class.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 2. Distribution des classes par split
# --------------------------------------------------------------------------
def plot_class_distribution(samples, seed=SEED):
    labels = [s[1] for s in samples]
    tr, va, te = _stratified_split(labels, seed=seed)

    def counts(idx):
        c = defaultdict(int)
        for i in idx:
            c[labels[i]] += 1
        return [c[k] for k in range(len(CLASSES))]

    data = {"train": counts(tr), "val": counts(va), "test": counts(te)}
    colors = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}

    x = range(len(CLASSES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (split, vals) in enumerate(data.items()):
        offs = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offs, vals, width, label=split, color=colors[split])
        ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("nombre d'echantillons")
    ax.set_title("Distribution des classes par split (stratifie, seed=%d)" % seed)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "class_distribution.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 3. Flux des formes de tenseur dans le CNN
# --------------------------------------------------------------------------
def plot_architecture(model):
    """Trace les formes (canaux x longueur) apres chaque etape cle,
    en sondant reellement le modele avec un tenseur factice."""
    # Sonde couche par couche pour recuperer les formes reelles.
    steps = [("entree", (N_CHANNELS, SEQ_LEN))]
    x = torch.zeros(1, N_CHANNELS, SEQ_LEN)
    labels_map = {
        0: "Conv1d 3->16",  3: "MaxPool /2",
        4: "Conv1d 16->32", 7: "MaxPool /2",
        8: "Conv1d 32->64", 11: "AvgPool ->1",
    }
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            x = layer(x)
            if i in labels_map:
                steps.append((labels_map[i], tuple(x.shape[1:])))
    steps.append(("Linear ->4", (len(CLASSES),)))

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    n = len(steps)
    box_w, gap = 1.0, 0.55
    max_ch = max(s[1][0] for s in steps)

    for i, (name, shape) in enumerate(steps):
        cx = i * (box_w + gap)
        ch = shape[0]
        length = shape[1] if len(shape) > 1 else 1
        # hauteur ~ canaux (normalisee), largeur du bloc constante
        h = 0.4 + 2.6 * (ch / max_ch)
        y0 = (3.2 - h) / 2
        rect = mpatches.FancyBboxPatch(
            (cx, y0), box_w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor="#333",
            facecolor="#cfe3f5" if len(shape) > 1 else "#f5d6cf")
        ax.add_patch(rect)
        shp = f"{ch}x{length}" if len(shape) > 1 else f"{ch}"
        ax.text(cx + box_w / 2, y0 + h + 0.12, name,
                ha="center", va="bottom", fontsize=8.5, rotation=0)
        ax.text(cx + box_w / 2, y0 + h / 2, shp,
                ha="center", va="center", fontsize=9, weight="bold")
        if i < n - 1:
            nx = (i + 1) * (box_w + gap)
            ax.annotate("", xy=(nx, 1.6), xytext=(cx + box_w, 1.6),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=1.3))

    ax.set_xlim(-0.3, n * (box_w + gap))
    ax.set_ylim(-0.2, 4.0)
    ax.set_title("MovementCNN -- flux des formes (canaux x longueur)  "
                 "[9 540 parametres]", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "architecture.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Chargement des echantillons...")
    samples = _load_samples(DATA_CSV)
    print(f"  {len(samples)} echantillons charges.")

    outs = [
        plot_signals_per_class(samples),
        plot_class_distribution(samples),
        plot_architecture(MovementCNN()),
    ]
    print("\nFigures ecrites :")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
