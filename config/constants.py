# config/constants.py

class Config:

    MODEL_REGISTRY = {
    "yolov11": {
        "module": "src.services.objectdetectors.yolov11.yolo_detector",
        "class": "YoloLoader",
        "default_weights": "src.services/objectdetectors/yolov11/models/yolo11n_custom.pt"
    },
    "mobilenetv3": {
        "module": "detectors.mobilenetv3.mobilenetv3_loader",
        "class": "ModelLoader",
        "default_weights": "detectors/mobilenetv3/model_files/model.pth"
    }
}