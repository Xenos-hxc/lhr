
from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/data/workspace/lhr/yolomv2/runs/multi/yolopm-all-cat/weights/best.pt')  # Validate the model
model.predict(source='/data/workspace/lhr/yolomv2/val', imgsz=(384,672), device=[3],name='test', save=True, conf=0.25, iou=0.45, show_labels=False)
