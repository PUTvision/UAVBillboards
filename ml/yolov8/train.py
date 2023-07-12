from ultralytics import YOLO


model = YOLO("yolov8l-seg.pt")

model.train(data="uavb.yaml", epochs=100, imgsz=960, batch=10)

model.val(split='test')
