from ultralytics import YOLO


model = YOLO("runs/segment/train14/weights/best.pt")

model.val(split='test')
