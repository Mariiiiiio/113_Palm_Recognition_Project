from ultralytics import YOLO
if __name__ == "__main__":
    # Build a YOLOv9c model from scratch
    # model = YOLO("yolov9c.yaml")

    # Build a YOLOv9c model from pretrained weight
    model = YOLO("yolov9t.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="data.yaml", epochs=600, imgsz=640, batch=16, workers=8, device=0, verbose=True, save=True)
    
    # Evaluate the model's performance on the validation set
    results = model.val()
    
    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    results = model("datasets/test/images")