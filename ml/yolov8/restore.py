from ultralytics import YOLO


model = YOLO("runs/segment/train13/weights/last.pt")

model.train(data="uavb.yaml", epochs=6, imgsz=960, batch=14)

model.val(split='test')
