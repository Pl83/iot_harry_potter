#!/usr/bin/env python3
"""Modele: CNN 1D pour classer les mouvements IoT.

Entree  : tenseur (B, 3, 100) -- canaux (ax, ay, az) x 100 pas de temps.
Sortie  : logits (B, 4) -- classes circle=0, horizontal=1, static=2, vertical=3.
          Logits BRUTS (pas de softmax) : CrossEntropyLoss l'attend ainsi.

Architecture (lame legere, adaptee aux ~1617 echantillons de train) :
    Conv1d(3->16, k5) -> BN -> ReLU -> MaxPool(2)          (B,16,50)
    Conv1d(16->32,k5) -> BN -> ReLU -> MaxPool(2)          (B,32,25)
    Conv1d(32->64,k3) -> BN -> ReLU -> AdaptiveAvgPool(1)  (B,64,1)
    Flatten -> Dropout(0.3) -> Linear(64->4)              (B,4)

AdaptiveAvgPool ecrase le temps : tete decouplee de la longueur, moins de
parametres (~15k), donc moins de sur-apprentissage.
"""
import torch
import torch.nn as nn

N_CHANNELS = 3
SEQ_LEN = 100
N_CLASSES = 4


class MovementCNN(nn.Module):
    """CNN 1D compact : (B,3,100) -> logits (B,4)."""

    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                 # 100 -> 50

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                 # 50 -> 25

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),         # 25 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B,64,1) -> (B,64)
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_params(model):
    """Nombre de parametres entrainables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _smoke_test():
    torch.manual_seed(0)
    model = MovementCNN()
    xb = torch.randn(8, N_CHANNELS, SEQ_LEN)
    out = model(xb)

    print("=" * 56)
    print("  MovementCNN  --  smoke test")
    print("=" * 56)
    print(model)
    print(f"\nEntree : {tuple(xb.shape)}")
    print(f"Sortie : {tuple(out.shape)}  dtype={out.dtype}")
    print(f"Parametres entrainables : {count_params(model):,}")

    assert out.shape == (8, N_CLASSES), "forme de sortie inattendue"
    assert out.dtype == torch.float32
    print("\n" + "=" * 56)
    print("  OK -- modele operationnel.")
    print("=" * 56)


if __name__ == "__main__":
    _smoke_test()
