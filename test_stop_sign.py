from ultralytics import YOLO
import cv2

model = YOLO('models/yolov8n.pt')

results = model('stop_sign.png')


for r in results:
    im_array = r.plot() 
    cv2.imshow("YOLOv8 Test", im_array)
    cv2.imwrite("result_stop_sign.jpg", im_array)

# cv2.waitKey(0)
# cv2.destroyAllWindows()