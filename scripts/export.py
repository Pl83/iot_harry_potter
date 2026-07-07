#!/usr/bin/env python3
"""Export : convertit le MovementCNN entraine au format ONNX.

Charge le meilleur checkpoint (models/movement_cnn.pt), fige le modele en
eval (BatchNorm/Dropout deterministes) et exporte vers models/movement_cnn.onnx.

  - Entree  'input'  : (batch, 3, 100) float32 -- axe batch DYNAMIQUE.
  - Sortie  'logits' : (batch, 4)     float32 -- logits BRUTS (pas de softmax).
    La classe = argmax(logits). softmax n'est pas necessaire (monotone).

Verification : rejoue le split test avec onnxruntime et compare les logits a
PyTorch (ecart max) + verifie l'egalite des predictions (argmax).

Usage:
    python scripts/export.py
"""
from pathlib import Path

import numpy as np
import torch

from dataloader import make_loaders, CLASSES
from model import MovementCNN, N_CHANNELS, SEQ_LEN

MODEL_DIR = Path(r"C:/Users/pierr/dev/iot/models")
CKPT_PATH = MODEL_DIR / "movement_cnn.pt"
ONNX_PATH = MODEL_DIR / "movement_cnn.onnx"

SEED = 42
OPSET = 17


def export_onnx(model):
    """Exporte le modele en eval vers ONNX avec un axe batch dynamique."""
    model.eval()
    dummy = torch.randn(1, N_CHANNELS, SEQ_LEN)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )


def verify(model):
    """Compare ONNX (onnxruntime) vs PyTorch sur tout le split test."""
    import onnxruntime as ort

    model.eval()
    _, _, test_loader = make_loaders(batch_size=64, seed=SEED)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    max_abs_diff = 0.0
    n, n_agree = 0, 0
    with torch.no_grad():
        for xb, _ in test_loader:
            torch_logits = model(xb).numpy()
            onnx_logits = sess.run(
                ["logits"], {"input": xb.numpy().astype(np.float32)})[0]
            max_abs_diff = max(max_abs_diff,
                               float(np.abs(torch_logits - onnx_logits).max()))
            n_agree += int((torch_logits.argmax(1) == onnx_logits.argmax(1)).sum())
            n += xb.shape[0]
    return max_abs_diff, n_agree, n


def main():
    device = torch.device("cpu")  # export & verif sur CPU : reproductible
    if not CKPT_PATH.exists():
        raise SystemExit(f"ERREUR: {CKPT_PATH} introuvable. Lancer train.py d'abord.")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    model = MovementCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])

    print("=" * 60)
    print(f"  EXPORT ONNX  —  checkpoint epoch {ckpt.get('epoch')} "
          f"(val acc {ckpt.get('val_acc'):.3f})")
    print("=" * 60)

    export_onnx(model)
    print(f"\nModele exporte : {ONNX_PATH}")
    print(f"  entree 'input'  : (batch, {N_CHANNELS}, {SEQ_LEN}) float32")
    print(f"  sortie 'logits' : (batch, {len(CLASSES)}) float32  "
          f"[classes: {CLASSES}]")

    max_diff, n_agree, n = verify(model)
    print(f"\nVerification (onnxruntime vs PyTorch sur {n} ech. test) :")
    print(f"  ecart max des logits   : {max_diff:.2e}")
    print(f"  predictions identiques : {n_agree}/{n}")

    assert max_diff < 1e-4, "ecart ONNX/PyTorch trop grand : export suspect"
    assert n_agree == n, "predictions divergentes : export defaillant"

    print("\n" + "=" * 60)
    print("  OK — ONNX fidele a PyTorch. Modele pret au deploiement.")
    print("=" * 60)


if __name__ == "__main__":
    main()
