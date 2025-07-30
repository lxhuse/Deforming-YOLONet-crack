from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v8/yolov8-seg-DSC.yaml").train(data="DSC_demo.yaml", epochs=100, imgsz=512, batch=4, save_period=10)
