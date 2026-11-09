"""
inspect_dataset.py
------------------
Module d'exploration et d'inspection du dataset de détection de véhicules.
Conforme à la Phase 2 & Phase 12 de RoadVision AI.
"""

from pathlib import Path
from collections import Counter
import argparse
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
    "car", "garbagevan", "human hauler", "minibus", "minivan",
    "motorbike", "pickup", "policecar", "rickshaw", "scooter",
    "suv", "taxi", "three wheelers -CNG-", "truck", "van", "wheelbarrow"
]


def list_dataset_files(data_dir: Path) -> dict:
    """
    Recense tous les fichiers d'images et d'annotations d'un dossier.
    """
    data_dir = Path(data_dir)
    images = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [p for p in data_dir.rglob("*.txt")]
    yaml_files = [p for p in data_dir.rglob("*.yaml")] + [p for p in data_dir.rglob("*.yml")]

    return {
        "images": images,
        "labels": labels,
        "yaml_files": yaml_files
    }


def count_images_by_split(data_dir: Path, splits=("train", "valid", "test")) -> pd.DataFrame:
    """
    Compte les images et annotations par split.
    """
    data_dir = Path(data_dir)
    records = []

    for split in splits:
        img_dir = data_dir / "images" / split if (data_dir / "images" / split).exists() else data_dir / split / "images"
        lbl_dir = data_dir / "labels" / split if (data_dir / "labels" / split).exists() else data_dir / split / "labels"

        nb_imgs = len([p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]) if img_dir.exists() else 0
        nb_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        records.append({
            "split": split,
            "images_dir_exists": img_dir.exists(),
            "labels_dir_exists": lbl_dir.exists(),
            "images_count": nb_imgs,
            "labels_count": nb_lbls
        })

    return pd.DataFrame(records)


def inspect_annotations(labels_dir: Path) -> dict:
    """
    Examine les annotations YOLO dans un dossier:
    fichiers vides, format invalide, bornes de coordonnées.
    """
    labels_dir = Path(labels_dir)
    label_files = list(labels_dir.glob("*.txt"))

    empty_files = []
    invalid_format_files = []
    out_of_bounds_files = []
    valid_boxes_count = 0

    for lbl in label_files:
        content = lbl.read_text(encoding="utf-8").strip()
        if not content:
            empty_files.append(lbl)
            continue

        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_format_files.append((lbl, idx, "Champs != 5"))
                continue

            try:
                cid, x, y, w, h = map(float, parts)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    out_of_bounds_files.append((lbl, idx, f"Coordonnées hors bornes: x={x}, y={y}, w={w}, h={h}"))
                else:
                    valid_boxes_count += 1
            except ValueError:
                invalid_format_files.append((lbl, idx, "Valeurs non numériques"))

    return {
        "total_files": len(label_files),
        "empty_files": empty_files,
        "invalid_format": invalid_format_files,
        "out_of_bounds": out_of_bounds_files,
        "valid_boxes_count": valid_boxes_count
    }


def summarize_classes(labels_dir: Path, class_names=CLASS_NAMES) -> pd.DataFrame:
    """
    Compte les occurrences de chaque classe présente dans les annotations.
    """
    labels_dir = Path(labels_dir)
    counter = Counter()

    for lbl in labels_dir.glob("*.txt"):
        content = lbl.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cid = int(float(parts[0]))
                    counter[cid] += 1
                except ValueError:
                    pass

    records = []
    for cid in range(len(class_names)):
        records.append({
            "class_id": cid,
            "class_name": class_names[cid] if cid < len(class_names) else f"classe_{cid}",
            "instances": counter.get(cid, 0)
        })

    df = pd.DataFrame(records)
    return df.sort_values(by="instances", ascending=False).reset_index(drop=True)


def validate_dataset_structure(data_dir: Path) -> bool:
    """
    Vérifie la structure complète de l'arborescence du dataset.
    """
    data_dir = Path(data_dir)
    expected_splits = ["train", "valid", "test"]

    print(f"--- Validation de la structure dans : {data_dir} ---")
    valid = True

    for split in expected_splits:
        img_p = data_dir / "images" / split
        lbl_p = data_dir / "labels" / split
        img_ok = img_p.is_dir()
        lbl_ok = lbl_p.is_dir()

        print(f"[{split.upper()}] Images: {'OK' if img_ok else 'MANQUANT'} | Labels: {'OK' if lbl_ok else 'MANQUANT'}")
        if not (img_ok and lbl_ok):
            valid = False

    return valid


def main():
    parser = argparse.ArgumentParser(description="Inspection du dataset RoadVision AI")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Chemin du dossier dataset")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    print("=" * 60)
    print(f"Inspection du dataset : {data_path.resolve()}")
    print("=" * 60)

    if not data_path.exists():
        print(f"Dossier {data_path} inexistant.")
        return

    validate_dataset_structure(data_path)

    df_splits = count_images_by_split(data_path)
    print("\n--- Répartition par split ---")
    print(df_splits.to_string(index=False))

    train_labels = data_path / "labels" / "train"
    if train_labels.exists():
        print("\n--- Inspection des annotations Train ---")
        anno_info = inspect_annotations(train_labels)
        print(f"Total fichiers labels : {anno_info['total_files']}")
        print(f"Fichiers vides : {len(anno_info['empty_files'])}")
        print(f"Erreurs de format : {len(anno_info['invalid_format'])}")
        print(f"Coordonnées hors bornes : {len(anno_info['out_of_bounds'])}")
        print(f"Boîtes valides : {anno_info['valid_boxes_count']}")

        print("\n--- Top classes dans Train ---")
        df_classes = summarize_classes(train_labels)
        print(df_classes.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

