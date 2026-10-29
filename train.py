"""Train one of the two models and copy the best weights into models/.

Examples:
    python train.py --task damage --data datasets/cardd/data.yaml
    python train.py --task parts  --data datasets/car-parts/data.yaml --epochs 80
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from app.config import MODELS_DIR, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["parts", "damage"], required=True)
    p.add_argument("--data", required=True, help="path to the dataset's data.yaml (YOLO format)")
    p.add_argument("--base", default="yolo11n-seg.pt", help="starting weights, e.g. yolo11s-seg.pt for more accuracy")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    args = p.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(ROOT / "runs"), name=args.task, exist_ok=True,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    target = MODELS_DIR / f"{args.task}.pt"
    shutil.copy(Path(model.trainer.best), target)
    print(f"Saved best weights to {target}")

    metrics = model.val()
    print(f"box mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    if hasattr(metrics, "seg"):
        print(f"mask mAP50: {metrics.seg.map50:.3f}  mAP50-95: {metrics.seg.map:.3f}")


if __name__ == "__main__":
    main()
