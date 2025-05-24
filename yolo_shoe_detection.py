from ultralytics import YOLO
from PIL import Image
import os

model = YOLO('best_yolo_weights.pt')

def get_yolo_cropped_images(img):
    img = img.convert("RGB")
    os.makedirs('temp_yolo_dir', exist_ok=True)
    img.save("temp_yolo_dir/temp_yolo_img.jpg")

    results = model('temp_yolo_dir/temp_yolo_img.jpg', conf=0.5)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return [img]

    cropped_list = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # coords from YOLO
        cropped = img.crop((x1, y1, x2, y2))
        cropped_list.append(cropped)
    
    return cropped_list