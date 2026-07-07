#!/usr/bin/env python3
"""Train / validation : boucle d'entrainement du MovementCNN.

Fait combattre ensemble les pieces deja forgees :
  - make_loaders (dataloader.py) : train/val/test.
  - MovementCNN (model.py)        : le modele (logits bruts).
  - build_training (train_setup.py) : criterion / optimizer / scheduler.

Strategie (decision Master, 2026-07-07) :
  - Budget 100 epochs + EARLY STOPPING (patience=15 sur la val loss).
  - On sauvegarde le MEILLEUR modele (val loss min) dans models/movement_cnn.pt
    (state_dict + meta : epoch, val_loss, val_acc, classes).
  - Device auto : CUDA si dispo, sinon CPU.
  - Graine fixe : reproductibilite.

Produit aussi figures/training_curves.png (loss & accuracy train/val vs epoch).

Usage:
    python scripts/train.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend fichier
import matplotlib.pyplot as plt

import torch

from dataloader import make_loaders, CLASSES
from model import MovementCNN
from train_setup import build_training

MODEL_DIR = Path(r"C:/Users/pierr/dev/iot/models")
FIG_DIR = Path(r"C:/Users/pierr/dev/iot/figures")
CKPT_PATH = MODEL_DIR / "movement_cnn.pt"

SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15          # epochs sans amelioration de val loss avant arret
BATCH_SIZE = 32


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """Une passe complete. Si optimizer fourni -> entrainement, sinon eval.

    Retourne (loss_moyenne, accuracy) ponderees par le nombre d'echantillons.
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train_mode:
                optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if train_mode:
                loss.backward()
                optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_correct += (out.argmax(1) == yb).sum().item()
            total_n += bs

    return total_loss / total_n, total_correct / total_n


def plot_curves(history, out_path):
    """Courbes d'apprentissage : loss (gauche) et accuracy (droite)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best = history["best_epoch"]

    fig, (axl, axa) = plt.subplots(1, 2, figsize=(13, 5))

    axl.plot(epochs, history["train_loss"], color="#4c72b0", label="train")
    axl.plot(epochs, history["val_loss"], color="#dd8452", label="val")
    axl.axvline(best, color="#d62728", linestyle="--", linewidth=1.3,
                label=f"meilleur (epoch {best})")
    axl.set_xlabel("epoch")
    axl.set_ylabel("loss (CrossEntropy)")
    axl.set_title("Loss train / val")
    axl.legend()

    axa.plot(epochs, history["train_acc"], color="#4c72b0", label="train")
    axa.plot(epochs, history["val_acc"], color="#dd8452", label="val")
    axa.axvline(best, color="#d62728", linestyle="--", linewidth=1.3,
                label=f"meilleur (epoch {best})")
    axa.set_xlabel("epoch")
    axa.set_ylabel("accuracy")
    axa.set_ylim(0, 1.02)
    axa.set_title("Accuracy train / val")
    axa.legend()

    fig.suptitle("MovementCNN — courbes d'apprentissage", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  TRAIN / VALIDATION  —  device={device}")
    print("=" * 60)

    train_loader, val_loader, _ = make_loaders(batch_size=BATCH_SIZE, seed=SEED)
    model = MovementCNN().to(device)
    criterion, optimizer, scheduler = build_training(model)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device,
                                     optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        lr = optimizer.param_groups[0]["lr"]
        marker = ""
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = "  <- meilleur"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:3d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
              f"lr {lr:.1e}{marker}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping : {PATIENCE} epochs sans progres. "
                  f"Arret a l'epoch {epoch}.")
            break

    history["best_epoch"] = best_epoch

    # Sauvegarde du MEILLEUR modele (pas le dernier).
    torch.save({
        "state_dict": best_state,
        "epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": history["val_acc"][best_epoch - 1],
        "classes": CLASSES,
    }, CKPT_PATH)

    fig_path = FIG_DIR / "training_curves.png"
    plot_curves(history, fig_path)

    print("\n" + "=" * 60)
    print(f"  Meilleur modele : epoch {best_epoch}  "
          f"val loss {best_val_loss:.4f}  "
          f"val acc {history['val_acc'][best_epoch - 1]:.3f}")
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  Courbes    : {fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
