from ultralytics import YOLO


if __name__ == "__main__":
    # Load a pre-trained YOLOv10n model
    model = YOLO("yolov10n.pt")
    # Load YOLOv10n model from scratch
    # model = YOLO("yolov10m.yaml")

    # Display model information (optional)
    model.info()

    # Train the model
    results = model.train(data="yolo_dataset_blended_N10GM\dataset.yaml", epochs=150, imgsz=1024, batch=16, workers=8, device=0, verbose=True, save=True,patience=50)
    
    # Evaluate the model's performance on the validation set
    results = model.val()
    
    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    results = model("yolo_dataset_blended_N10GM/test/images")