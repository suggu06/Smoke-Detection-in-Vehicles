# src/evaluate.py
from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def validate_yolo(args):
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split="val",
        save_json=True  # saves COCO-style JSON
    )
    # Ultralytics returns a rich object; pull the key metrics
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(metrics.box.f1),
        "num_images": int(metrics.speed["n"]),
    }
    print("\n📊 Validation summary:")
    for k, v in summary.items():
        print(f"{k:>10}: {v}")

    # Save summary CSV next to weights
    out_dir = Path(metrics.save_dir)
    pd.DataFrame([summary]).to_csv(out_dir / "val_summary.csv", index=False)
    print(f"\n📝 Saved metrics to: {out_dir/'val_summary.csv'}")

# --- Optional: frame-level “smoke present?” accuracy for slides ---
# Defines positive if any detection of class 'smoke' exists in a frame at conf>=threshold.
def frame_level_metrics(args):
    """
    Computes confusion matrix on the VAL split by collapsing detections per-image
    into a binary label: smoke-present vs no-smoke. Ground truth is derived from
    YOLO label files (>=1 line means smoke present).
    """
    from ultralytics import YOLO
    import cv2, os
    import numpy as np
    from glob import glob

    model = YOLO(args.weights)
    img_paths = sorted(glob(str(Path(args.val_images) / "*.*")))
    y_true, y_pred = [], []

    for img_path in img_paths:
        # GT: label file with same stem
        stem = Path(img_path).stem
        label_file = Path(args.val_labels) / f"{stem}.txt"
        has_gt_smoke = label_file.exists() and os.path.getsize(label_file) > 0
        y_true.append(1 if has_gt_smoke else 0)

        # Prediction: does model return any 'smoke' boxes above conf?
        preds = model.predict(img_path, conf=args.conf, iou=args.iou, verbose=False)
        det = preds[0].boxes
        has_pred_smoke = (det is not None) and (len(det) > 0)
        y_pred.append(1 if has_pred_smoke else 0)

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    report = classification_report(y_true, y_pred, target_names=["smoke","no-smoke"], digits=4, zero_division=0)
    print("\n🧾 Frame-level metrics (binary):")
    print("Confusion Matrix [rows: true smoke/no-smoke, cols: pred smoke/no-smoke]\n", cm)
    print("\nClassification Report:\n", report)

    # Save CSV
    df = pd.DataFrame({"image": [Path(p).name for p in img_paths], "gt": y_true, "pred": y_pred})
    out_dir = Path("runs/frame_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "frame_metrics.csv", index=False)
    print(f"Saved per-frame results: {out_dir/'frame_metrics.csv'}")
