import cv2
import numpy as np

def detect_obstacle(
    frame,
    roi_y_start=0.25,          # nhìn phần dưới ảnh (55% -> 100%)
    roi_x_left=0.2,            # nhìn giữa ảnh
    roi_x_right=0.8,
    blur_ksize=5,              # giảm nhiễu
    canny1=60,                 # Canny thresholds (tune)
    canny2=160,
    edge_ratio_threshold=0.06, # % pixel là biên đủ lớn -> obstacle
    min_contour_area=800       # lọc rác nhỏ (tune theo độ phân giải)
):
    """
    Returns: (is_obstacle: bool, score: float, debug: dict)
    score = max(edge_ratio, contour_area_ratio)
    """
    if frame is None or frame.size == 0:
        return False, 0.0, {}

    h, w = frame.shape[:2]
    y0 = int(h * roi_y_start)
    x1 = int(w * roi_x_left)
    x2 = int(w * roi_x_right)
    roi = frame[y0:h, x1:x2]
    if roi.size == 0:
        return False, 0.0, {}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, canny1, canny2)
    edge_ratio = float((edges > 0).mean())

    # Contour check: obstacle thường tạo mảng biên lớn
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(roi.shape[0] * roi.shape[1])

    max_area = 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a > max_area:
            max_area = a

    contour_area_ratio = float(max_area / roi_area) if roi_area > 0 else 0.0

    # Điều kiện obstacle: hoặc edge_ratio đủ lớn, hoặc có contour đủ lớn
    is_obstacle = (edge_ratio >= edge_ratio_threshold) or (max_area >= min_contour_area)

    score = max(edge_ratio, contour_area_ratio)
    debug = {
        "edge_ratio": edge_ratio,
        "max_contour_area": max_area,
        "contour_area_ratio": contour_area_ratio,
        "roi_shape": roi.shape,
    }
    return is_obstacle, score, debug


# ====== DEBOUNCE CLASS ======
class ObstacleStopper:
    def __init__(self, require_consecutive=3):
        self.require_consecutive = require_consecutive
        self.count = 0

    def update(self, is_obstacle: bool) -> bool:
        if is_obstacle:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.require_consecutive


# ====== MAIN LOOP ======
# def main():
#     # --- Open camera ---
#     cap = cv2.VideoCapture(0)  # Pi camera: thường là 0
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     if not cap.isOpened():
#         print("❌ Cannot open camera")
#         return

#     print("✅ Camera opened")

#     stopper = ObstacleStopper(require_consecutive=3)
#     stopped = False

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("⚠️ Failed to grab frame")
#                 continue

#             # --- Detect obstacle ---
#             is_obstacle, score, debug = detect_obstacle(frame)

#             # --- Decision ---
#             if stopper.update(is_obstacle):
#                 if not stopped:
#                     car_stop()
#                     stopped = True
#             else:
#                 if stopped:
#                     car_forward()
#                     stopped = False

#             # --- Optional debug ---
#             print(
#                 f"Obstacle={is_obstacle} | "
#                 f"score={score:.3f} | "
#                 f"edge_ratio={debug.get('edge_ratio', 0):.3f}"
#             )

#             time.sleep(0.03)  # ~30 FPS, giảm tải CPU cho Pi

#     except KeyboardInterrupt:
#         print("\n🛑 Stop by user")

#     finally:
#         car_stop()
#         cap.release()
#         cv2.destroyAllWindows()
#         print("✅ Camera released, car stopped")


import cv2
from pathlib import Path


def main():
    IMAGE_PATH = Path(__file__).parent / "r2.jpg"

    if not IMAGE_PATH.exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        return

    frame = cv2.imread(str(IMAGE_PATH))
    if frame is None:
        print("❌ Failed to read image")
        return

    print(f"✅ Loaded image: {IMAGE_PATH}")
    print(f"Image shape: {frame.shape}")

    # ---- Detect obstacle ----
    is_obstacle, score, debug = detect_obstacle(frame)

    # ---- Result ----
    print("\n=== DETECTION RESULT ===")
    print(f"Obstacle detected : {is_obstacle}")
    print(f"Score             : {score:.4f}")

    for k, v in debug.items():
        print(f"{k:20s}: {v}")

    # ---- (Optional) Visual debug ----
    # Vẽ ROI + edges để nhìn trực quan
    h, w = frame.shape[:2]
    y0 = int(h * 0.55)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)

    roi = frame[y0:h, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    cv2.rectangle(frame, (x1, y0), (x2, h), (0, 255, 0), 2)

    cv2.imshow("Original + ROI", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Edges", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
