#!/usr/bin/env python3
"""Criterion / Optim : composants d'entrainement du modele.

Assemble les trois pieces que la boucle d'entrainement consommera :
  - criterion : nn.CrossEntropyLoss (attend des LOGITS bruts, comme le modele).
  - optimizer : Adam (lr=1e-3, weight_decay=1e-4).
  - scheduler : ReduceLROnPlateau sur la val loss (divise le lr quand ca stagne).

Choix (decision Master, 2026-07-07) :
  - PAS de ponderation de classe : le split train est quasi equilibre
    (circle=401, horizontal=396, static=426, vertical=394 ; max/min=1.08).
  - weight_decay=1e-4 : L2 legere, en renfort du Dropout(0.3) deja present,
    double rempart anti-overfit sur seulement 1617 echantillons de train.
  - ReduceLROnPlateau plutot que cosine : reactif, pas besoin de figer le
    nombre d'epochs a l'avance.

Usage:
    from train_setup import build_training
    criterion, optimizer, scheduler = build_training(model)
    ...
    scheduler.step(val_loss)   # a appeler apres chaque epoch de validation
"""
import torch
import torch.nn as nn


def build_training(model, lr=1e-3, weight_decay=1e-4,
                   scheduler_factor=0.5, scheduler_patience=5):
    """Construit (criterion, optimizer, scheduler) pour `model`.

    Args:
        model: le nn.Module a entrainer.
        lr: pas d'apprentissage initial d'Adam.
        weight_decay: coefficient L2 (regularisation).
        scheduler_factor: facteur de reduction du lr sur plateau.
        scheduler_patience: nombre d'epochs sans amelioration avant reduction.

    Returns:
        (criterion, optimizer, scheduler)
        scheduler surveille une metrique a MINIMISER (val loss) : appeler
        `scheduler.step(val_loss)` apres chaque validation.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=scheduler_factor, patience=scheduler_patience)
    return criterion, optimizer, scheduler


def _smoke_test():
    """Prouve que le trio optimise : sur-apprend un seul batch aleatoire,
    la loss doit s'effondrer et l'accuracy monter a 1.0."""
    from model import MovementCNN, N_CHANNELS, SEQ_LEN, N_CLASSES

    torch.manual_seed(0)
    model = MovementCNN()
    criterion, optimizer, scheduler = build_training(model)

    print("=" * 56)
    print("  TRAIN_SETUP  --  smoke test (overfit d'un batch)")
    print("=" * 56)
    print(f"criterion : {criterion.__class__.__name__}")
    print(f"optimizer : {optimizer.__class__.__name__} "
          f"(lr={optimizer.param_groups[0]['lr']}, "
          f"weight_decay={optimizer.param_groups[0]['weight_decay']})")
    print(f"scheduler : {scheduler.__class__.__name__} "
          f"(factor={scheduler.factor}, patience={scheduler.patience})")

    # Un seul batch fixe, labels aleatoires : le modele doit le memoriser.
    xb = torch.randn(16, N_CHANNELS, SEQ_LEN)
    yb = torch.randint(0, N_CLASSES, (16,))

    model.train()
    first = None
    for step in range(200):
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        if first is None:
            first = loss.item()
        if step % 40 == 0 or step == 199:
            acc = (out.argmax(1) == yb).float().mean().item()
            print(f"  step {step:3d} : loss={loss.item():.4f}  acc={acc:.2f}")

    final = loss.item()
    acc = (model(xb).argmax(1) == yb).float().mean().item()
    print(f"\nloss {first:.4f} -> {final:.4f}   acc finale={acc:.2f}")
    assert final < first * 0.2, "la loss ne descend pas : optimizer defaillant"
    assert acc == 1.0, "le batch n'est pas memorise : machinerie suspecte"
    print("\n" + "=" * 56)
    print("  OK -- criterion/optimizer/scheduler operationnels.")
    print("=" * 56)


if __name__ == "__main__":
    _smoke_test()
