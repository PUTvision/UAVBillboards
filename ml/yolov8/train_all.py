from ultralytics import YOLO


model = YOLO("yolov8x-seg.pt")

model.train(data="uavb.yaml", epochs=300, imgsz=960, batch=10)
