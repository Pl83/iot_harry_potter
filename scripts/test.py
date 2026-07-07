#!/usr/bin/env python3
"""Test : jugement final du MovementCNN sur le split test (jamais vu).

Charge le meilleur checkpoint (models/movement_cnn.pt), passe sur le split test
(seed=42 -> split identique a l'entrainement) et rend le verdict :
  - accuracy globale ;
  - precision / recall / f1 PAR CLASSE (calcul maison depuis la matrice de
    confusion, aucune dependance sklearn).

Produit 2 figures dans figures/ :
  1. confusion_matrix.png  -- heatmap 4x4 (vrai vs predit).
  2. per_class_metrics.png -- barres precision/recall/f1 par classe.

Usage:
    python scripts/test.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend fichier
import matplotlib.pyplot as plt
import numpy as np

import torch

from dataloader import make_loaders, CLASSES
from model import MovementCNN, N_CLASSES

MODEL_DIR = Path(r"C:/Users/pierr/dev/iot/models")
FIG_DIR = Path(r"C:/Users/pierr/dev/iot/figures")
CKPT_PATH = MODEL_DIR / "movement_cnn.pt"

SEED = 42
BATCH_SIZE = 32


def collect_predictions(model, loader, device):
    """Renvoie (y_true, y_pred) en numpy int, sur tout le loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            y_pred.append(out.argmax(1).cpu())
            y_true.append(yb)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def confusion_matrix(y_true, y_pred, n_classes=N_CLASSES):
    """Matrice n x n : lignes = vraie classe, colonnes = classe predite."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    """precision, recall, f1 par classe depuis la matrice de confusion."""
    tp = np.diag(cm).astype(float)
    pred_tot = cm.sum(axis=0).astype(float)   # colonnes = predits
    true_tot = cm.sum(axis=1).astype(float)   # lignes = vrais
    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    recall = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def plot_confusion(cm, out_path):
    """Heatmap annotee. Diagonale = bons classements."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nb echantillons")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("classe predite")
    ax.set_ylabel("classe vraie")

    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "#222",
                    fontsize=11, weight="bold")

    acc = np.diag(cm).sum() / cm.sum()
    ax.set_title(f"Matrice de confusion — test (accuracy {acc:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_per_class(precision, recall, f1, out_path):
    """Barres groupees precision/recall/f1 par classe."""
    x = np.arange(len(CLASSES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, precision, width, label="precision", color="#4c72b0")
    b2 = ax.bar(x, recall, width, label="recall", color="#dd8452")
    b3 = ax.bar(x + width, f1, width, label="f1", color="#55a868")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Metriques par classe — test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CKPT_PATH.exists():
        raise SystemExit(f"ERREUR: {CKPT_PATH} introuvable. Lancer train.py d'abord.")
    ckpt = torch.load(CKPT_PATH, map_location=device)

    model = MovementCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, _, test_loader = make_loaders(batch_size=BATCH_SIZE, seed=SEED)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = per_class_metrics(cm)
    acc = (y_true == y_pred).mean()

    print("=" * 60)
    print(f"  TEST  —  checkpoint epoch {ckpt.get('epoch')} "
          f"(val acc {ckpt.get('val_acc'):.3f})")
    print("=" * 60)
    print(f"\nEchantillons test : {len(y_true)}")
    print(f"Accuracy globale  : {acc:.4f}\n")
    print(f"{'classe':<12} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    support = cm.sum(axis=1)
    for i, c in enumerate(CLASSES):
        print(f"{c:<12} {precision[i]:6.3f} {recall[i]:7.3f} "
              f"{f1[i]:6.3f} {support[i]:8d}")
    print(f"\n{'macro-moy':<12} {precision.mean():6.3f} "
          f"{recall.mean():7.3f} {f1.mean():6.3f}")

    cm_path = FIG_DIR / "confusion_matrix.png"
    pc_path = FIG_DIR / "per_class_metrics.png"
    plot_confusion(cm, cm_path)
    plot_per_class(precision, recall, f1, pc_path)

    print("\nFigures ecrites :")
    print(f"  {cm_path}")
    print(f"  {pc_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
