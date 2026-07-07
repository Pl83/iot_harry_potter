#!/usr/bin/env python3
"""Verif dataset: controle d'integrite complet du dataset unifie.

Verifie: coherence manifest<->disque, en-tete/lignes/colonnes uniformes,
absence de NaN/inf, sanite de la normalisation (moyenne~0, sigma~1 hors
colonnes constantes), distribution classes x groupes, drapeaux z_imputed.
"""
import csv
import math
from pathlib import Path
from collections import defaultdict

DST = Path(r"C:/Users/pierr/dev/iot/dataset")
CANON = ["ax", "ay", "az"]
CLASSES = ["circle", "horizontal", "static", "vertical"]


def main():
    errors, warnings = [], []

    # 1. Charger le manifest
    with (DST / "manifest.csv").open(newline="") as f:
        manifest = list(csv.DictReader(f))
    man_paths = {m["path"] for m in manifest}

    # 2. Fichiers sur disque vs manifest
    disk = {str(p.relative_to(DST)).replace("\\", "/")
            for p in DST.rglob("*.csv") if p.name != "manifest.csv"}
    orphans_disk = disk - man_paths
    orphans_man = man_paths - disk
    for o in sorted(orphans_disk):
        errors.append(f"Sur disque mais absent du manifest: {o}")
    for o in sorted(orphans_man):
        errors.append(f"Dans manifest mais absent du disque: {o}")

    # 3/4. Controle de chaque fichier
    cls_count = defaultdict(int)
    grp_count = defaultdict(int)
    xtab = defaultdict(lambda: defaultdict(int))  # group -> class -> n
    z_imputed_n = 0
    bad_norm = 0
    for m in manifest:
        p = DST / m["path"]
        cls_count[m["label"]] += 1
        grp_count[m["group"]] += 1
        xtab[m["group"]][m["label"]] += 1
        if m["z_imputed"] == "1":
            z_imputed_n += 1
        with p.open(newline="") as f:
            r = csv.reader(f)
            header = next(r)
            rows = [row for row in r if row]
        if header != CANON:
            errors.append(f"En-tete != {CANON}: {m['path']} -> {header}")
        if len(rows) != 100:
            errors.append(f"{len(rows)} lignes (attendu 100): {m['path']}")
        # colonnes: numerique, fini, normalisation
        cols = list(zip(*[[float(v) for v in row] for row in rows]))
        for ci, col in enumerate(cols):
            if any(math.isnan(v) or math.isinf(v) for v in col):
                errors.append(f"NaN/inf col {CANON[ci]}: {m['path']}")
                continue
            n = len(col)
            mean = sum(col) / n
            sigma = math.sqrt(sum((v - mean) ** 2 for v in col) / n)
            is_const = all(v == 0.0 for v in col)
            if is_const:
                continue  # colonne constante attendue (ex. az impute)
            if abs(mean) > 1e-4 or abs(sigma - 1.0) > 1e-4:
                bad_norm += 1
                if bad_norm <= 10:
                    warnings.append(
                        f"Norme suspecte {CANON[ci]} (moy={mean:.2e}, "
                        f"sig={sigma:.4f}): {m['path']}")

    # === RAPPORT ===
    print("=" * 60)
    print(f"  VERIF DATASET  —  {len(manifest)} echantillons")
    print("=" * 60)

    print("\nDistribution par classe:")
    for c in CLASSES:
        print(f"  {c:12s}: {cls_count[c]:4d}")
    print(f"  {'TOTAL':12s}: {sum(cls_count.values()):4d}")

    print("\nDistribution par groupe:")
    for g in sorted(grp_count):
        print(f"  {g:28s}: {grp_count[g]:4d}")

    print("\nTableau croise groupe x classe:")
    hdr = f"  {'GROUPE':28s} " + " ".join(f"{c:>9s}" for c in CLASSES)
    print(hdr)
    for g in sorted(xtab):
        line = f"  {g:28s} " + " ".join(f"{xtab[g][c]:>9d}" for c in CLASSES)
        print(line)

    print(f"\nZ impute (Tom, az=0): {z_imputed_n} fichiers")
    print(f"Colonnes a normalisation suspecte: {bad_norm}")

    print("\n" + "=" * 60)
    if errors:
        print(f"  ECHEC — {len(errors)} erreur(s):")
        for e in errors[:30]:
            print("   X", e)
    else:
        print("  AUCUNE ERREUR — dataset conforme.")
    if warnings:
        print(f"\n  {len(warnings)} avertissement(s) (echantillon):")
        for w in warnings[:10]:
            print("   !", w)
    print("=" * 60)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
