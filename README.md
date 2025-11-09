Vehicle Smoke Emission Detection System

A computer vision-based system designed to detect vehicles emitting smoke using YOLO-based object detection. The project aims to assist pollution control authorities by identifying smoke-emitting vehicles from video feeds, supporting greener and sustainable transportation.

Week 1 Progress

Finalized the project idea and problem statement

Selected the tech stack: Python, YOLOv8, OpenCV

Created initial folder structure and GitHub repository

Added basic YOLO setup for vehicle detection (to be enhanced in Week 2)

Tech Stack

Python

YOLOv8

OpenCV

NumPy

Upcoming (Week 2 Plan)

Integrate smoke detection logic

Implement alert logging system for identified violations

Smoke-Detection-in-Vehicles

A real-time system to detect smoke emissions from vehicles using computer vision.
This repository provides code to train, run inference, and evaluate a YOLOv8-based model on a custom smoke-detection dataset in YOLO format.

Week 2 Milestone

Dataset prepared, annotations organized, initial training completed, and inference tested on sample images.
The repository includes scripts and steps to reproduce training and inference results.

Key Features

YOLOv8-based object detection for vehicles and smoke

Structured project layout with dedicated scripts for training, inference, and evaluation

Works on both CPU and GPU environments

Fully reproducible setup using requirements.txt

Command-line and Python script-based execution support


Project Structure 
Smoke-Detection-in-Vehicles/
│
├── src/
│   ├── train.py        # Training wrapper for Ultralytics YOLO
│   ├── infer.py        # Run inference on images/videos
│   ├── evaluate.py     # Model validation and metric evaluation
│   └── __init__.py
│
├── datasets/
│   └── smoke/          # Custom dataset (YOLO format)
│       ├── data.yaml
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/       # Optional
│           ├── images/
│           └── labels/
│
├── results/            # Training and inference outputs
├── README.md
└── requirements.txt


Running Inference

Run prediction on a single image or an entire folder:
yolo predict model="runs/train/<exp>/weights/best.pt" \
     source="path/to/image_or_folder" save=True


Model Validation
Using YOLO CLI:
yolo task=detect mode=val model="runs/train/<exp>/weights/best.pt" \
     data=datasets/smoke/data.yaml
Using script:
python src/evaluate.py --weights runs/train/<exp>/weights/best.pt \
    --data datasets/smoke/data.yaml


Results (Sample Outputs)
results/
├── sample_train_batch.jpg
└── sample_prediction.jpg



![vehicle_detection_sample jpg](https://github.com/user-attachments/assets/d905315c-d602-4aef-9ffa-04786beffe6c)








