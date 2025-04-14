from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
if __name__ == "__main__":
    model = YOLO("runs/detect/train5/weights/best.pt")


    # Validate the model
        # metrics = model.val()
        # print(metrics.box.map)

    # Predict with the model
    results = model("test_blended", save=True)  # predict on an image