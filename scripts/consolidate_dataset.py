#!/usr/bin/env python3
"""Consolidation: fusionne tous les echantillons du dataset/ en UN seul CSV
au format long.

Chaque echantillon (100 pas de temps x 3 canaux) devient 100 lignes:
    sample_id, t, ax, ay, az, label, group

- sample_id: entier stable (ordre du manifest: label, group, orig_name).
- t: index temporel 0..99.
- ax,ay,az: valeurs deja normalisees (z-score/fichier) lues depuis dataset/.
- label: classe (circle|horizontal|static|vertical).
- group: groupe d'origine, conserve POUR AUDIT (non utilise pour le split).

Sortie: dataset/dataset.csv
"""
import csv
from pathlib import Path

DST = Path(r"C:/Users/pierr/dev/iot/dataset")
OUT = DST / "dataset.csv"
CANON = ["ax", "ay", "az"]


def main():
    manifest_path = DST / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"ERREUR: {manifest_path} introuvable. Lancer build_dataset.py d'abord.")

    with manifest_path.open(newline="") as f:
        manifest = list(csv.DictReader(f))

    n_rows_written = 0
    with OUT.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["sample_id", "t", "ax", "ay", "az", "label", "group"])

        for sample_id, m in enumerate(manifest):
            path = DST / m["path"]
            label, group = m["label"], m["group"]
            with path.open(newline="") as fin:
                r = csv.reader(fin)
                header = next(r)
                if header != CANON:
                    raise SystemExit(f"En-tete inattendu {header} dans {path}")
                rows = [row for row in r if row]
            if len(rows) != 100:
                raise SystemExit(f"{len(rows)} lignes (attendu 100) dans {path}")
            for t, (ax, ay, az) in enumerate(rows):
                w.writerow([sample_id, t, ax, ay, az, label, group])
                n_rows_written += 1

    print(f"OK: {len(manifest)} echantillons -> {OUT}")
    print(f"  {n_rows_written} lignes de donnees ({len(manifest)} x 100)")


if __name__ == "__main__":
    main()
