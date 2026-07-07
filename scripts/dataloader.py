#!/usr/bin/env python3
"""Dataloader PyTorch pour le dataset de mouvements IoT.

Lit le CSV unique consolide (dataset/dataset.csv, format long) et fournit
des DataLoader train/val/test.

- Chaque echantillon -> tenseur float32 de forme (3, 100) : canaux-d'abord
  (ax, ay, az) x 100 pas de temps. Convention Conv1d (batch, channels, length).
- Labels: circle=0, horizontal=1, static=2, vertical=3.
- Split ALEATOIRE stratifie par classe, graine fixe (reproductible).
  NB (decision Master): PAS de split par groupe. Risque de fuite de signature
  capteur assume -- cible = modele deploye sur ces memes capteurs.

Usage:
    from scripts.dataloader import make_loaders
    train, val, test = make_loaders(batch_size=32)
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

DATA_CSV = Path(r"C:/Users/pierr/dev/iot/dataset/dataset.csv")
CLASSES = ["circle", "horizontal", "static", "vertical"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
SEQ_LEN = 100
N_CHANNELS = 3
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test


def _load_samples(csv_path: Path):
    """Reconstruit les echantillons depuis le CSV long.

    Retourne: liste de tuples (x, y, group)
      x: tensor float32 (3, 100), y: int, group: str.
    """
    if not csv_path.exists():
        raise SystemExit(
            f"ERREUR: {csv_path} introuvable. Lancer consolidate_dataset.py d'abord.")

    # sample_id -> {t: (ax, ay, az)}, + label/group
    channels = defaultdict(dict)
    meta = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            sid = int(row["sample_id"])
            t = int(row["t"])
            channels[sid][t] = (
                float(row["ax"]), float(row["ay"]), float(row["az"]))
            if sid not in meta:
                meta[sid] = (row["label"], row["group"])

    samples = []
    for sid in sorted(channels):
        ts = channels[sid]
        if len(ts) != SEQ_LEN:
            raise SystemExit(f"Echantillon {sid}: {len(ts)} pas (attendu {SEQ_LEN})")
        label, group = meta[sid]
        # (3, 100): ligne 0=ax, 1=ay, 2=az ; colonnes = temps.
        x = torch.empty((N_CHANNELS, SEQ_LEN), dtype=torch.float32)
        for t in range(SEQ_LEN):
            ax, ay, az = ts[t]
            x[0, t], x[1, t], x[2, t] = ax, ay, az
        samples.append((x, LABEL2IDX[label], group))
    return samples


def _stratified_split(labels, ratios=SPLIT_RATIOS, seed=SEED):
    """Indices train/val/test, aleatoires mais stratifies par classe."""
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for y in sorted(by_class):
        idx = by_class[y][:]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_idx += idx[:n_train]
        val_idx += idx[n_train:n_train + n_val]
        test_idx += idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


class MovementDataset(Dataset):
    """Sous-ensemble en memoire d'echantillons (x (3,100), y)."""

    def __init__(self, samples, indices):
        self._data = [(samples[i][0], samples[i][1]) for i in indices]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        x, y = self._data[i]
        return x, y


def make_loaders(batch_size=32, csv_path=DATA_CSV, seed=SEED):
    """Construit (train_loader, val_loader, test_loader).

    Prechargement complet en memoire (~11 MB), aucun I/O par batch.
    num_workers=0 (robustesse Windows).
    """
    samples = _load_samples(csv_path)
    labels = [s[1] for s in samples]
    train_idx, val_idx, test_idx = _stratified_split(labels, seed=seed)

    train_ds = MovementDataset(samples, train_idx)
    val_ds = MovementDataset(samples, val_idx)
    test_ds = MovementDataset(samples, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)
    return train_loader, val_loader, test_loader


def train_class_counts(csv_path=DATA_CSV, seed=SEED):
    """Effectifs par classe dans le split train (utile a l'etape Criterion)."""
    samples = _load_samples(csv_path)
    labels = [s[1] for s in samples]
    train_idx, _, _ = _stratified_split(labels, seed=seed)
    counts = defaultdict(int)
    for i in train_idx:
        counts[labels[i]] += 1
    return {CLASSES[k]: counts[k] for k in sorted(counts)}


def _smoke_test():
    """Controle de sanite: tailles, distribution, forme d'un batch."""
    samples = _load_samples(DATA_CSV)
    labels = [s[1] for s in samples]
    train_idx, val_idx, test_idx = _stratified_split(labels)

    print("=" * 56)
    print(f"  DATALOADER  —  {len(samples)} echantillons")
    print("=" * 56)

    def dist(idx):
        c = defaultdict(int)
        for i in idx:
            c[labels[i]] += 1
        return " ".join(f"{CLASSES[k]}={c[k]}" for k in sorted(c))

    print(f"\nSplit (seed={SEED}, ratios={SPLIT_RATIOS}):")
    print(f"  train: {len(train_idx):4d}   {dist(train_idx)}")
    print(f"  val  : {len(val_idx):4d}   {dist(val_idx)}")
    print(f"  test : {len(test_idx):4d}   {dist(test_idx)}")

    # Verifs: partition complete, aucun recouvrement.
    allidx = train_idx + val_idx + test_idx
    assert len(allidx) == len(samples), "partition incomplete"
    assert len(set(allidx)) == len(samples), "recouvrement entre splits"

    train_loader, val_loader, test_loader = make_loaders(batch_size=32)
    xb, yb = next(iter(train_loader))
    print(f"\nBatch train: x={tuple(xb.shape)} dtype={xb.dtype}  "
          f"y={tuple(yb.shape)} dtype={yb.dtype}")
    assert xb.shape[1:] == (N_CHANNELS, SEQ_LEN), "forme de batch inattendue"
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64

    print(f"\nEffectifs train par classe: {train_class_counts()}")
    print("\n" + "=" * 56)
    print("  OK — dataloader operationnel.")
    print("=" * 56)


if __name__ == "__main__":
    _smoke_test()
