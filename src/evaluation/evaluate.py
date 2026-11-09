"""
evaluate.py
-----------
Module d'évaluation quantitative du modèle sur l'ensemble de test.
Conforme à la Phase 9, Phase 11 & Phase 12 de RoadVision AI.
"""

from pathlib import Path
import json
import argparse
import pandas as pd

CLASS_NAMES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
    "car", "garbagevan", "human hauler", "minibus", "minivan",
    "motorbike", "pickup", "policecar", "rickshaw", "scooter",
    "suv", "taxi", "three wheelers -CNG-", "truck", "van", "wheelbarrow"
]


def load_model(weights_path: Path):
    """
    Charge le modèle YOLO à partir des poids spécifiés.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids introuvable : {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Le package 'ultralytics' est requis pour évaluer le modèle. Exécutez : pip install ultralytics")

    print(f"Chargement du modèle depuis : {weights_path}")
    return YOLO(str(weights_path))


def evaluate_test_set(
    model,
    data_yaml: Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    split: str = "test"
):
    """
    Lance l'évaluation du modèle sur le split spécifié (test ou val).
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Configuration data.yaml introuvable : {data_yaml}")

    print(f"Évaluation sur le split '{split}' avec data={data_yaml}, device={device}...")
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True
    )
    return metrics


def extract_metrics(results) -> dict:
    """
    Extrait et formate les métriques globales :
    Precision, Recall, mAP50, mAP50-95, et temps d'inférence.
    """
    box = results.box
    speed = results.speed

    precision = float(box.mp)
    recall = float(box.mr)
    map50 = float(box.map50)
    map50_95 = float(box.map)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    metrics_dict = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "speed_ms": {
            "preprocess": round(speed.get("preprocess", 0.0), 2),
            "inference": round(speed.get("inference", 0.0), 2),
            "loss": round(speed.get("loss", 0.0), 2),
            "postprocess": round(speed.get("postprocess", 0.0), 2),
            "total": round(sum(speed.values()), 2)
        }
    }
    return metrics_dict


def summarize_per_class(results, class_names=CLASS_NAMES) -> pd.DataFrame:
    """
    Extrait les performances (Precision, Recall, mAP50, mAP50-95) par catégorie de véhicule.
    """
    records = []
    box = results.box

    maps_per_class = box.maps if hasattr(box, "maps") and box.maps is not None else []
    p_per_class = box.p if hasattr(box, "p") and box.p is not None else []
    r_per_class = box.r if hasattr(box, "r") and box.r is not None else []

    num_classes = len(class_names)
    for i in range(num_classes):
        c_name = class_names[i]
        c_p = float(p_per_class[i]) if i < len(p_per_class) else 0.0
        c_r = float(r_per_class[i]) if i < len(r_per_class) else 0.0
        c_map50 = float(maps_per_class[i]) if i < len(maps_per_class) else 0.0

        records.append({
            "class_id": i,
            "class_name": c_name,
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "mAP50": round(c_map50, 4)
        })

    df = pd.DataFrame(records)
    return df.sort_values(by="mAP50", ascending=False).reset_index(drop=True)


def save_metrics(metrics: dict, output_path: Path):
    """
    Sauvegarde les métriques au format JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Métriques sauvegardées dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation de modèle RoadVision AI")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Chemin des poids du modèle")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Chemin de data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split d'évaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, 0, etc.)")
    parser.add_argument("--output", type=str, default="results/metrics/test_metrics.json", help="Chemin de sortie")
    args = parser.parse_args()

    weights_p = Path(args.weights)
    if not weights_p.exists():
        fallback = Path("yolov9c.pt")
        if fallback.exists():
            weights_p = fallback
        else:
            print(f"Attention : aucun fichier de poids trouvé à '{args.weights}'.")
            return

    model = load_model(weights_p)
    results = evaluate_test_set(model, Path(args.data), device=args.device, split=args.split)

    metrics = extract_metrics(results)
    print("\n" + "=" * 50)
    print("MÉTRIQUES GLOBALES DE TEST")
    print("=" * 50)
    for k, v in metrics.items():
        if k != "speed_ms":
            print(f"{k.upper():<12} : {v}")
    print(f"Temps total / image : {metrics['speed_ms']['total']} ms")

    df_classes = summarize_per_class(results)
    print("\n--- Bilan par classe (Top 10) ---")
    print(df_classes.head(10).to_string(index=False))

    save_metrics(metrics, Path(args.output))

    csv_out = Path(args.output).with_suffix(".csv")
    df_classes.to_csv(csv_out, index=False)
    print(f"Détail par classe sauvegardé dans : {csv_out}")


if __name__ == "__main__":
    main()

