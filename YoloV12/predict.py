from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
if __name__ == "__main__":
    
    model = YOLO("C:/Users/USER/Desktop/University/Project/Palm_Project/yolov12/runs/detect/train/weights/best.pt")
    #model = YOLO("C:/Users/USER/Desktop/University/Project/Palm_Project/yolov11/runs/detect/train7_nir_first/weights/best.pt")

    # Validate the model
        # metrics = model.val()
        # print(metrics.box.map)

    # Predict with the model
    results = model("test_blended", save=True)  # predict on an image
    # results = model("test_nir", save=True)  # predict on an image
    #results = model("test_vis", save=True)  # predict on an image
 