#!/usr/bin/env python3
"""Export des poids du MovementCNN vers un fichier JS pour l'inference navigateur.

Le moniteur web (web/mpu6050-monitor.html) reimplemente le forward pass en JS
pur : il lui faut les poids. On les serialise en une liste de couches ordonnee
(conv / bn / relu / maxpool / avgpool / linear) dans :

    web/movement_cnn_weights.js   ->   window.MODEL_WEIGHTS = {...};

Charge par <script src>, ce qui fonctionne en file:// (contrairement a fetch).

Verification : un forward pass numpy qui consomme EXACTEMENT la liste exportee
(rounding compris) est compare au modele PyTorch sur des entrees aleatoires.
Ce numpy est le miroir fidele du JS -> s'il concorde, le JS aussi.

Usage:
    python scripts/export_web_weights.py
"""
import json
from math import floor, log10
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import MovementCNN, N_CHANNELS, SEQ_LEN
from dataloader import CLASSES

MODEL_DIR = Path(r"C:/Users/pierr/dev/iot/models")
CKPT_PATH = MODEL_DIR / "movement_cnn.pt"
OUT_PATH = Path(r"C:/Users/pierr/dev/iot/web/movement_cnn_weights.js")

SIG = 6  # chiffres significatifs conserves (compacite sans perte de prediction)


def _sig(x):
    """Arrondit un scalaire a SIG chiffres significatifs."""
    if x == 0.0:
        return 0.0
    return round(x, SIG - 1 - floor(log10(abs(x))))


def _round(obj):
    """Arrondit recursivement une liste imbriquee de floats."""
    if isinstance(obj, list):
        return [_round(o) for o in obj]
    return _sig(float(obj))


def build_layers(model):
    """Serialise features + classifier en liste de couches ordonnee."""
    layers = []
    for m in model.features:
        if isinstance(m, nn.Conv1d):
            layers.append({
                "type": "conv",
                "weight": _round(m.weight.tolist()),   # (out, in, k)
                "bias": _round(m.bias.tolist()),
                "pad": int(m.padding[0]),
            })
        elif isinstance(m, nn.BatchNorm1d):
            layers.append({
                "type": "bn",
                "weight": _round(m.weight.tolist()),
                "bias": _round(m.bias.tolist()),
                "mean": _round(m.running_mean.tolist()),
                "var": _round(m.running_var.tolist()),
                "eps": float(m.eps),
            })
        elif isinstance(m, nn.ReLU):
            layers.append({"type": "relu"})
        elif isinstance(m, nn.MaxPool1d):
            layers.append({"type": "maxpool", "size": int(m.kernel_size)})
        elif isinstance(m, nn.AdaptiveAvgPool1d):
            layers.append({"type": "avgpool"})
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            layers.append({
                "type": "linear",
                "weight": _round(m.weight.tolist()),   # (out, in)
                "bias": _round(m.bias.tolist()),
            })
    return layers


def forward_np(x, layers):
    """Miroir numpy du forward JS. x: (3, 100). Renvoie les logits (4,)."""
    for L in layers:
        t = L["type"]
        if t == "conv":
            W = np.array(L["weight"]); b = np.array(L["bias"]); pad = L["pad"]
            xin = np.pad(x, ((0, 0), (pad, pad)))
            out_c, in_c, k = W.shape
            lout = xin.shape[1] - k + 1
            y = np.zeros((out_c, lout))
            for oc in range(out_c):
                acc = np.zeros(lout)
                for ic in range(in_c):
                    for kk in range(k):
                        acc += W[oc, ic, kk] * xin[ic, kk:kk + lout]
                y[oc] = acc + b[oc]
            x = y
        elif t == "bn":
            w = np.array(L["weight"]); b = np.array(L["bias"])
            m = np.array(L["mean"]); v = np.array(L["var"]); eps = L["eps"]
            x = (x - m[:, None]) / np.sqrt(v[:, None] + eps) * w[:, None] + b[:, None]
        elif t == "relu":
            x = np.maximum(x, 0.0)
        elif t == "maxpool":
            s = L["size"]; lout = x.shape[1] // s
            x = x[:, :lout * s].reshape(x.shape[0], lout, s).max(axis=2)
        elif t == "avgpool":
            x = x.mean(axis=1, keepdims=True)
        elif t == "linear":
            W = np.array(L["weight"]); b = np.array(L["bias"])
            x = W @ x.reshape(-1) + b
    return x


def main():
    if not CKPT_PATH.exists():
        raise SystemExit(f"ERREUR: {CKPT_PATH} introuvable. Lancer train.py d'abord.")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = MovementCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    layers = build_layers(model)

    # --- Verification de parite (numpy miroir du JS vs PyTorch) ---
    print("=" * 60)
    print("  EXPORT POIDS WEB  --  verification de parite")
    print("=" * 60)
    torch.manual_seed(1)
    max_diff, mism = 0.0, 0
    N = 50
    for _ in range(N):
        xt = torch.randn(1, N_CHANNELS, SEQ_LEN)
        with torch.no_grad():
            ref = model(xt).numpy()[0]
        got = forward_np(xt.numpy()[0], layers)
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        if ref.argmax() != got.argmax():
            mism += 1
    print(f"  entrees testees        : {N}")
    print(f"  ecart max des logits   : {max_diff:.2e}")
    print(f"  predictions divergentes: {mism}/{N}")
    assert mism == 0, "le forward JS-miroir diverge de PyTorch : export invalide"
    assert max_diff < 1e-3, "ecart trop grand (rounding trop agressif ?)"

    obj = {
        "classes": CLASSES,
        "n_channels": N_CHANNELS,
        "seq_len": SEQ_LEN,
        "layers": layers,
        "meta": {"epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc")},
    }
    js = "window.MODEL_WEIGHTS = " + json.dumps(obj, separators=(",", ":")) + ";\n"
    OUT_PATH.write_text(js, encoding="utf-8")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\n  Ecrit : {OUT_PATH}  ({size_kb:.0f} Ko)")
    print("=" * 60)
    print("  OK -- poids exportes, forward JS-miroir fidele a PyTorch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
