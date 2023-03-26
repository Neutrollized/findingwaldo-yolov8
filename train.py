from ultralytics import YOLO

# Load the pre-trained model.
model = YOLO('pretrained_models/yolov8n.pt')

# Training - https://docs.ultralytics.com/modes/train/
results = model.train(
   data='waldo.yaml',
   imgsz=1024,
   epochs=30,
   batch=4,
   name='yolov8n_30e_b4'
)
