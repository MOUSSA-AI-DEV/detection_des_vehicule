"""
split_dataset.py
----------------
Module de partitionnement et de préparation des ensembles Train, Validation et Test.
Conforme à la Phase 5 & Phase 12 de RoadVision AI.
"""

from pathlib import Path
import shutil
import argparse
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TARGET_COUNTS = {
    "train": 2297,
    "valid": 298,
    "test": 403
}


def collect_valid_samples(images_dir: Path, labels_dir: Path, exclude_empty: bool = True) -> list[tuple[Path, Path]]:
    """
    Associe chaque image à son annotation si les deux existent et sont valides.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    pairs = []
    if not images_dir.exists() or not labels_dir.exists():
        return pairs

    for img_p in images_dir.iterdir():
        if img_p.is_file() and img_p.suffix.lower() in IMAGE_EXTENSIONS:
            lbl_p = labels_dir / f"{img_p.stem}.txt"
            if lbl_p.is_file():
                if exclude_empty and not lbl_p.read_text(encoding="utf-8").strip():
                    continue
                pairs.append((img_p, lbl_p))

    return pairs


def split_samples(
    train_source_pairs: list[tuple[Path, Path]],
    valid_source_pairs: list[tuple[Path, Path]],
    test_size: int = 403,
    train_target_size: int = 2297,
    valid_target_size: int = 298,
    random_state: int = 42
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Applique le protocole RoadVision AI :
    1. Prélèvement du split test (403 images) depuis train_source_pairs avec random_state=42.
    2. Sélection de 2297 paires pour train_final.
    3. Sélection de 298 paires pour valid_final.
    """
    if len(train_source_pairs) <= test_size:
        test_size = max(1, int(len(train_source_pairs) * 0.15))

    train_remaining, test_pairs = train_test_split(
        train_source_pairs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    if len(train_remaining) > train_target_size:
        train_final, _ = train_test_split(
            train_remaining,
            train_size=train_target_size,
            random_state=random_state,
            shuffle=True
        )
    else:
        train_final = train_remaining

    if len(valid_source_pairs) > valid_target_size:
        valid_final, _ = train_test_split(
            valid_source_pairs,
            train_size=valid_target_size,
            random_state=random_state,
            shuffle=True
        )
    else:
        valid_final = valid_source_pairs

    return {
        "train": train_final,
        "valid": valid_final,
        "test": test_pairs
    }


def copy_split_files(splits: dict[str, list[tuple[Path, Path]]], output_dir: Path):
    """
    Copie les images et labels dans l'arborescence standard data/processed/.
    """
    output_dir = Path(output_dir)

    for split_name, pairs in splits.items():
        img_dest = output_dir / "images" / split_name
        lbl_dest = output_dir / "labels" / split_name

        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, img_dest / img_p.name)
            shutil.copy2(lbl_p, lbl_dest / lbl_p.name)

        print(f"[{split_name.upper()}] : {len(pairs)} paires copiées vers {output_dir}")


def verify_splits(output_dir: Path) -> bool:
    """
    Contrôle les effectifs de chaque split et l'absence stricte de chevauchement (data leakage).
    """
    output_dir = Path(output_dir)
    splits = ["train", "valid", "test"]
    stems_per_split = {}
    is_ok = True

    print("\n--- Vérification de l'intégrité des splits ---")
    for s in splits:
        img_dir = output_dir / "images" / s
        lbl_dir = output_dir / "labels" / s

        imgs = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}

        if imgs != lbls:
            print(f"[ERREUR] Incohérence images/labels dans {s}: {len(imgs)} imgs vs {len(lbls)} labels")
            is_ok = False
        else:
            print(f"[{s.upper()}] : {len(imgs)} images parfaitement appariées.")

        stems_per_split[s] = imgs

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = stems_per_split[s1] & stems_per_split[s2]
            if overlap:
                print(f"[ALERTE FUITE DE DONNÉES] {len(overlap)} fichiers partagés entre {s1} et {s2} !")
                is_ok = False
            else:
                print(f"[OK] Aucun chevauchement entre {s1} et {s2}.")

    return is_ok


def main():
    parser = argparse.ArgumentParser(description="Partitionnement du dataset RoadVision AI")
    parser.add_argument("--source-dir", type=str, default="data/raw/trafic_data", help="Dossier brut contenant train et valid")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Dossier de destination")
    parser.add_argument("--test-size", type=int, default=403, help="Nombre d'images pour le split test")
    parser.add_argument("--random-state", type=int, default=42, help="Seed pour la reproductibilité")
    args = parser.parse_args()

    src_path = Path(args.source_dir)
    dest_path = Path(args.output_dir)

    print("=" * 60)
    print(f"Préparation des splits : {src_path} -> {dest_path}")
    print("=" * 60)

    train_imgs = src_path / "train" / "images"
    train_lbls = src_path / "train" / "labels"
    valid_imgs = src_path / "valid" / "images"
    valid_lbls = src_path / "valid" / "labels"

    train_pairs = collect_valid_samples(train_imgs, train_lbls)
    valid_pairs = collect_valid_samples(valid_imgs, valid_lbls)

    print(f"Paires collectées dans Train brut : {len(train_pairs)}")
    print(f"Paires collectées dans Valid brut : {len(valid_pairs)}")

    if not train_pairs:
        print("Aucune paire d'images trouvée. Vérifiez le chemin --source-dir.")
        return

    splits = split_samples(train_pairs, valid_pairs, test_size=args.test_size, random_state=args.random_state)
    copy_split_files(splits, dest_path)
    verify_splits(dest_path)


if __name__ == "__main__":
    main()

