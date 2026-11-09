"""
clean_dataset.py
----------------
Module de nettoyage et de validation des données d'images et d'annotations.
Conforme à la Phase 4 & Phase 12 de RoadVision AI.
"""

from pathlib import Path
import json
import argparse
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image(image_path: Path) -> tuple[bool, str]:
    """
    Vérifie qu'un fichier image est lisible, non vide et intègre.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        return False, "Fichier inexistant"

    if image_path.stat().st_size == 0:
        return False, "Taille nulle (0 octets)"

    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Dimensions invalides: {w}x{h}"
        return True, "OK"
    except Exception as e:
        return False, f"Image corrompue: {str(e)}"


def validate_label_file(label_path: Path, num_classes: int = 21) -> tuple[bool, str, list]:
    """
    Vérifie le format et les bornes d'un fichier d'annotation YOLO.
    Retourne (is_valid, reason, valid_boxes).
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        return False, "Fichier introuvable", []

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return False, "Label vide", []

    valid_boxes = []
    for line_idx, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            return False, f"Ligne {line_idx} invalide: champs != 5", []

        try:
            cid, x, y, w, h = map(float, parts)
        except ValueError:
            return False, f"Ligne {line_idx}: valeurs non numériques", []

        if not (0 <= cid < num_classes and cid.is_integer()):
            return False, f"Ligne {line_idx}: class_id {cid} hors bornes [0, {num_classes-1}]", []

        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False, f"Ligne {line_idx}: centre (x={x}, y={y}) hors limites [0, 1]", []

        if w <= 0 or h <= 0 or w > 1 or h > 1:
            return False, f"Ligne {line_idx}: dimensions invalides (w={w}, h={h})", []

        valid_boxes.append((int(cid), x, y, w, h))

    return True, "OK", valid_boxes


def match_images_and_labels(images_dir: Path, labels_dir: Path) -> dict:
    """
    Vérifie la correspondance 1-pour-1 entre images et annotations.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    images = {p.stem: p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}

    common_stems = set(images.keys()) & set(labels.keys())
    missing_labels = [images[s] for s in set(images.keys()) - set(labels.keys())]
    missing_images = [labels[s] for s in set(labels.keys()) - set(images.keys())]

    return {
        "matched_count": len(common_stems),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "matched_stems": common_stems
    }


def clean_dataset(raw_dir: Path, output_dir: Path = None, num_classes: int = 21) -> dict:
    """
    Parcourt le dataset, identifie les images et annotations invalides,
    et produit un bilan complet.
    """
    raw_dir = Path(raw_dir)
    report = {
        "initial_images": 0,
        "initial_labels": 0,
        "corrupted_images": [],
        "empty_labels": [],
        "invalid_labels": [],
        "unmatched_images": [],
        "unmatched_labels": [],
        "valid_pairs": []
    }

    images = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    report["initial_images"] = len(images)

    for img_p in images:
        is_img_ok, img_reason = validate_image(img_p)
        if not is_img_ok:
            report["corrupted_images"].append({"file": str(img_p), "reason": img_reason})
            continue

        expected_lbl = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        if not expected_lbl.exists():
            expected_lbl = img_p.with_suffix(".txt")

        if not expected_lbl.exists():
            report["unmatched_images"].append(str(img_p))
            continue

        is_lbl_ok, lbl_reason, boxes = validate_label_file(expected_lbl, num_classes=num_classes)
        if not is_lbl_ok:
            if "vide" in lbl_reason.lower():
                report["empty_labels"].append({"file": str(expected_lbl), "reason": lbl_reason})
            else:
                report["invalid_labels"].append({"file": str(expected_lbl), "reason": lbl_reason})
            continue

        report["valid_pairs"].append((str(img_p), str(expected_lbl)))

    report["valid_pairs_count"] = len(report["valid_pairs"])
    return report


def save_cleaning_report(report_dict: dict, output_path: Path):
    """
    Enregistre le bilan de nettoyage au format JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        k: v if not isinstance(v, list) or (v and not isinstance(v[0], tuple)) else [list(x) for x in v]
        for k, v in report_dict.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Bilan de nettoyage enregistré dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Nettoyage et validation du dataset RoadVision AI")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Chemin des données brutes")
    parser.add_argument("--output-report", type=str, default="results/cleaning_report.json", help="Chemin du rapport")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Dossier source introuvable : {data_path}")
        return

    print("=" * 60)
    print(f"Nettoyage du dataset : {data_path}")
    print("=" * 60)

    report = clean_dataset(data_path)
    print(f"Images initiales : {report['initial_images']}")
    print(f"Images corrompues : {len(report['corrupted_images'])}")
    print(f"Labels vides : {len(report['empty_labels'])}")
    print(f"Labels invalides : {len(report['invalid_labels'])}")
    print(f"Images orphelines (sans label) : {len(report['unmatched_images'])}")
    print(f"Paires valides retenues : {report['valid_pairs_count']}")

    save_cleaning_report(report, Path(args.output_report))


if __name__ == "__main__":
    main()

