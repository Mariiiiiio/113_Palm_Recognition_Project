from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
if __name__ == "__main__":
    # model = YOLO("C:/Users/USER/Desktop/University/Project/國科會計畫/yolov9/runs/detect/train11/weights/best.pt")
    model = YOLO("runs/detect/train/weights/best.pt")


    # Validate the model
        # metrics = model.val()
        # print(metrics.box.map)

    # Predict with the model
    results = model("datasets/test/images", save=True, save_txt=True)  # predict on an image