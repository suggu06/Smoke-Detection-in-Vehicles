# src/train.py
from ultralytics import YOLO

def train_yolo(args):
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True
    )
    print("\n✅ Training complete.")
    print(f"📂 Run dir: {results.save_dir}")
    print("Tip: best weights will be saved as best.pt inside the run directory.")
