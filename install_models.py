import os
from ultralytics import YOLO

# Tạo thư mục models nếu chưa có
os.makedirs('models', exist_ok=True)

# Tải model và lưu vào thư mục models
model = YOLO('yolov8n.pt') 
model.save('models/yolov8n.pt')
print("Đã tải xong model!")