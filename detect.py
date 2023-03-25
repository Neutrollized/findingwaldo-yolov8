# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
import cv2

model = YOLO('trained_models/best8n.pt')
results = model('waldo_test.jpg')  # predict on an image

# Working with Results - https://docs.ultralytics.com/modes/predict/#working-with-results
for result in results:
  print(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())

result_plotted = results[0].plot()
cv2.imwrite("result.jpg", result_plotted)
