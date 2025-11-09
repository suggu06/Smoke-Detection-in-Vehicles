import argparse
from pathlib import Path

# use relative imports because we run with: python -m src.main ...
from .train import train_yolo
from .evaluate import validate_yolo, frame_level_metrics
from .infer import run_inference



def parse_args():
    p = argparse.ArgumentParser("Smoke Detection – backend CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--data", type=str, default="datasets/smoke/data.yaml")
    p_train.add_argument("--model", type=str, default="yolov8n.pt")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--project", type=str, default="runs/train")
    p_train.add_argument("--name", type=str, default="smoke_yolov8n")

    # validate
    p_val = sub.add_parser("val")
    p_val.add_argument("--weights", type=str, required=True, help="path to best.pt")
    p_val.add_argument("--data", type=str, default="datasets/smoke/data.yaml")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--conf", type=float, default=0.25)
    p_val.add_argument("--iou", type=float, default=0.50)

    # frame-level “accuracy”
    p_frame = sub.add_parser("frame-metrics")
    p_frame.add_argument("--weights", type=str, required=True)
    p_frame.add_argument("--val_images", type=str, default="datasets/smoke/images/val")
    p_frame.add_argument("--val_labels", type=str, default="datasets/smoke/labels/val")
    p_frame.add_argument("--conf", type=float, default=0.25)
    p_frame.add_argument("--iou", type=float, default=0.5)

    # inference on a folder/video
    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--weights", type=str, required=True)
    p_inf.add_argument("--source", type=str, required=True, help="image/dir/video path")
    p_inf.add_argument("--conf", type=float, default=0.25)

    return p.parse_args()

def main():
    args = parse_args()
    if args.cmd == "train":
        train_yolo(args)
    elif args.cmd == "val":
        validate_yolo(args)
    elif args.cmd == "frame-metrics":
        frame_level_metrics(args)
    elif args.cmd == "infer":
        run_inference(args)

if __name__ == "__main__":
    main()
