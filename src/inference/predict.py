"""
predict.py
----------
Module d'inférence et de visualisation des détections sur de nouvelles images.
Conforme à la Phase 10 & Phase 12 de RoadVision AI.
"""

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
    "car", "garbagevan", "human hauler", "minibus", "minivan",
    "motorbike", "pickup", "policecar", "rickshaw", "scooter",
    "suv", "taxi", "three wheelers -CNG-", "truck", "van", "wheelbarrow"
]

COLORS = plt.cm.tab20(np.linspace(0, 1, len(CLASS_NAMES)))


def load_model(weights_path: Path):
    """
    Prépare et charge le modèle YOLO entraîné.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids introuvable : {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Le package 'ultralytics' est requis pour l'inférence. Exécutez : pip install ultralytics")

    return YOLO(str(weights_path))


def predict_image(
    model,
    image_path: Path,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu"
) -> list[dict]:
    """
    Effectue une détection sur une image donnée et retourne la liste des détections formatées.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"classe_{cls_id}")

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

    return detections


def filter_detections(detections: list[dict], conf_threshold: float = 0.25) -> list[dict]:
    """
    Filtre les détections selon un seuil de confiance minimum.
    """
    return [d for d in detections if d["confidence"] >= conf_threshold]


def draw_detections(
    image_path: Path,
    detections: list[dict],
    class_names: list[str] = CLASS_NAMES,
    output_path: Path = None,
    show: bool = False
):
    """
    Dessine les boîtes englobantes et étiquettes sur l'image avec Matplotlib / PIL.
    """
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        cid = det["class_id"]
        cname = det["class_name"]
        conf = det["confidence"]
        box_w = xmax - xmin
        box_h = ymax - ymin

        color = COLORS[cid % len(COLORS)]

        rect = patches.Rectangle(
            (xmin, ymin),
            box_w,
            box_h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        label_text = f"{cname} {conf:.2f}"
        ax.text(
            xmin,
            max(0, ymin - 5),
            label_text,
            color="white",
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.8, pad=2)
        )

    ax.axis("off")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"Image annotée sauvegardée : {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def save_prediction(output_path: Path, image_obj):
    """
    Enregistre l'image prédite.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_obj, Image.Image):
        image_obj.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Inférence RoadVision AI")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Poids du modèle")
    parser.add_argument("--source", type=str, required=True, help="Image ou dossier d'images à tester")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--output-dir", type=str, default="results/predictions", help="Dossier de sortie")
    parser.add_argument("--device", type=str, default="cpu", help="Périphérique (cpu, cuda, 0)")
    args = parser.parse_args()

    weights_p = Path(args.weights)
    if not weights_p.exists():
        fallback = Path("yolov9c.pt")
        if fallback.exists():
            weights_p = fallback
        else:
            print(f"Poids introuvables : {args.weights}")
            return

    model = load_model(weights_p)
    source_p = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_p.is_file():
        image_list = [source_p]
    elif source_p.is_dir():
        image_list = [p for p in source_p.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        print(f"Source introuvable : {source_p}")
        return

    print("=" * 60)
    print(f"Inférence sur {len(image_list)} image(s) avec modèle {weights_p.name} (seuil={args.conf})")
    print("=" * 60)

    for img_p in image_list:
        dets = predict_image(model, img_p, conf_threshold=args.conf, device=args.device)
        filtered = filter_detections(dets, conf_threshold=args.conf)
        print(f"{img_p.name} : {len(filtered)} véhicule(s) détecté(s)")
        for d in filtered:
            print(f"   -> {d['class_name']} ({d['confidence'] * 100:.1f}%) à {d['bbox']}")

        out_img = out_dir / f"pred_{img_p.name}"
        draw_detections(img_p, filtered, output_path=out_img)


if __name__ == "__main__":
    main()

