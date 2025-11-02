from ultralytics import YOLO

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded successfully!")
    print("This is the initial setup.")

if __name__ == "__main__":
    main()
