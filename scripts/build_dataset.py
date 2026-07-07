#!/usr/bin/env python3
"""Init dataset: fusionne les CSV de tous les groupes de la classe en un dataset
unifie, harmonise et normalise.

- Schema canonique: ax,ay,az (3 axes accelerometre).
  * Groupes 3 axes (Arthur, Clement, Ethan, Jason, MEHDI): 3 premieres colonnes.
  * Lucas (6 axes ax,ay,az,gx,gy,gz): on garde les 3 accelero, on jette le gyro.
  * Tom (2 axes X,Y): on impute az=0.0 (decision Master) -> marque z_imputed.
- Normalisation z-score par colonne, par fichier: N = (x - moyenne) / sigma.
  Sigma de population (ddof=0). Colonne constante (ex. az impute de Tom) -> 0.0.
- Sortie: dataset/<classe>/<groupe>__<fichier_origine>.csv  (+ manifest.csv)

Le nom de groupe prefixe chaque fichier pour permettre un split train/val
par personne (eviter la fuite de donnees entre train et validation).
"""
import csv
import math
import re
import sys
from pathlib import Path

SRC = Path(r"C:/Users/pierr/dev/iot/data/_extract")
DST = Path(r"C:/Users/pierr/dev/iot/dataset")
CANON = ["ax", "ay", "az"]
CLASSES = ["circle", "horizontal", "static", "vertical"]


def sanitize(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name)
    return re.sub(r"_+", "_", s).strip("_")


def zscore(col):
    """z-score population. Retourne colonne normalisee; constante -> zeros."""
    n = len(col)
    mean = sum(col) / n
    var = sum((v - mean) ** 2 for v in col) / n
    sigma = math.sqrt(var)
    # Colonne constante: sigma sous le bruit flottant relatif a l'echelle.
    if sigma <= 1e-9 * (abs(mean) + 1.0):
        return [0.0] * n
    return [(v - mean) / sigma for v in col]


def load_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header (ignore, on utilise l'ordre des colonnes)
        return [[float(v) for v in row] for row in reader if row]


def main():
    if not SRC.exists():
        sys.exit(f"ERREUR: {SRC} introuvable. Extraire les zips d'abord.")

    DST.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {c: 0 for c in CLASSES}

    for group_dir in sorted(d for d in SRC.iterdir() if d.is_dir()):
        group = sanitize(group_dir.name)
        for cls in CLASSES:
            cls_dir = group_dir / cls
            if not cls_dir.is_dir():
                continue
            out_cls = DST / cls
            out_cls.mkdir(parents=True, exist_ok=True)
            for src in sorted(cls_dir.glob("*.csv")):
                rows = load_rows(src)
                if not rows:
                    continue
                ncols = len(rows[0])
                z_imputed = ncols < 3
                # 3 premieres colonnes = accelerometre X,Y,Z (gyro vient apres).
                ax = [r[0] for r in rows]
                ay = [r[1] for r in rows]
                az = [r[2] for r in rows] if ncols >= 3 else [0.0] * len(rows)

                nax, nay, naz = zscore(ax), zscore(ay), zscore(az)

                out_name = f"{group}__{src.name}"
                out_path = out_cls / out_name
                with out_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(CANON)
                    for a, b, c in zip(nax, nay, naz):
                        w.writerow([f"{a:.6f}", f"{b:.6f}", f"{c:.6f}"])

                counts[cls] += 1
                manifest.append({
                    "path": str(out_path.relative_to(DST)).replace("\\", "/"),
                    "label": cls,
                    "group": group,
                    "n_rows": len(rows),
                    "n_channels": 3,
                    "src_axes": ncols,
                    "z_imputed": int(z_imputed),
                    "orig_name": src.name,
                })

    manifest.sort(key=lambda m: (m["label"], m["group"], m["orig_name"]))
    with (DST / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "label", "group", "n_rows", "n_channels",
            "src_axes", "z_imputed", "orig_name"])
        w.writeheader()
        w.writerows(manifest)

    total = sum(counts.values())
    print(f"OK: {total} echantillons ecrits dans {DST}")
    for c in CLASSES:
        print(f"  {c:12s}: {counts[c]}")
    print(f"Manifest: {DST / 'manifest.csv'} ({len(manifest)} lignes)")


if __name__ == "__main__":
    main()
