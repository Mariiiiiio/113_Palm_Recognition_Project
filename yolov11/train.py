from ultralytics import YOLO




if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # pass any model type
    
    results = model.train(data="yolo_dataset_blended_N10GM\dataset.yaml", epochs=150, imgsz=1024, batch=16, workers=8, device=0, verbose=True, save=True, patience=50)

    # Evaluate the model's performance on the validation set
    results = model.val()
    
    # Perform object detection on an image using the model
    results = model("yolo_dataset_Visible/test/images")