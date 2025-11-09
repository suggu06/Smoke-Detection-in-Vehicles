# src/infer.py
from ultralytics import YOLO

def run_inference(args):
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=True)
    print("\n🎯 Inference complete.")
    print(f"📂 Outputs saved under: {results[0].save_dir}")
