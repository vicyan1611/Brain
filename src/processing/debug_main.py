import cv2
import numpy as np
import time
import csv
import os

# --- IMPORT CÁC CLASS CỦA BẠN ---
# Đảm bảo bạn đã đặt đúng cấu trúc thư mục
from src.processing.utils.lane_detection import LaneCurveEstimator
from src.processing.utils.processPerception import AdaptiveController

# --- CẤU HÌNH ---
INPUT_VIDEO = 'src/processing/input.mp4'       # Đường dẫn video test của bạn
OUTPUT_VIDEO = 'src/processing/debug_output.mp4' # Video kết quả
CSV_LOG_FILE = 'src/processing/debug_data.csv'   # File log số liệu

def draw_dashboard(img, steer_deg, speed_pwm, offset, heading, curvature, target_speed_raw):
    """Vẽ thông số lên màn hình để dễ debug"""
    h, w = img.shape[:2]
    
    # 1. Vẽ nền bảng điều khiển
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # 2. Hiển thị thông số Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Cột trái: Thông số đầu vào (Perception)
    cv2.putText(img, f"Offset: {offset:.3f} m", (10, 20), font, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f"Heading: {np.rad2deg(heading):.2f} deg", (10, 40), font, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f"Radius: {curvature:.0f} m", (10, 60), font, 0.5, (0, 255, 255), 1)
    
    # Cột phải: Thông số đầu ra (Control)
    # Màu xanh lá: Bình thường, Màu đỏ: Cảnh báo (góc lái lớn/tốc độ thấp)
    color_steer = (0, 255, 0) if abs(steer_deg) < 15 else (0, 0, 255)
    color_speed = (0, 255, 0) if speed_pwm > 300 else (0, 165, 255)
    
    cv2.putText(img, f"STEER: {steer_deg}", (w - 150, 20), font, 0.6, color_steer, 2)
    cv2.putText(img, f"SPEED (PWM): {speed_pwm}", (w - 180, 50), font, 0.6, color_speed, 2)
    cv2.putText(img, f"(Target: {target_speed_raw:.1f})", (w - 180, 70), font, 0.4, (200, 200, 200), 1)

    # 3. Vẽ Vô lăng ảo (Steering Visualizer)
    center_x, center_y = w // 2, h - 30
    radius = 25
    # Vẽ vòng tròn vô lăng
    cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 2)
    # Tính tọa độ kim chỉ hướng
    angle_rad = np.deg2rad(steer_deg - 90) # -90 để 0 độ hướng lên trên
    end_x = int(center_x + radius * np.cos(angle_rad))
    end_y = int(center_y + radius * np.sin(angle_rad))
    cv2.line(img, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)

    return img

def main():
    # 1. Khởi tạo
    estimator = LaneCurveEstimator()
    controller = AdaptiveController()
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {INPUT_VIDEO}")
        return

    # Lấy thông số video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 480  # Resize về chuẩn
    height = 240
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # CSV Logger
    csv_file = open(CSV_LOG_FILE, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    # Header chi tiết
    csv_writer.writerow([
        "Frame", "Timestamp", 
        "Offset_m", "Heading_rad", "Heading_deg", "Curvature_m", 
        "Target_Speed_Raw", "Kp_Applied", # Thông số Controller
        "Final_Steer_Deg", "Final_Speed_PWM" # Thông số gửi xuống mạch
    ])

    print(f"Đang xử lý video... Log lưu tại {CSV_LOG_FILE}")
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = time.time() - start_time
        
        # Resize đúng chuẩn model
        frame = cv2.resize(frame, (width, height))
        
        # --- BƯỚC 1: PERCEPTION ---
        # Lấy thông số từ ảnh
        offset, curvature, heading, _ = estimator.process(frame)
        
        # --- BƯỚC 2: CONTROL ---
        # Lấy góc lái và tốc độ gốc
        steer_deg_raw, target_speed = controller.get_control(offset, heading)
        
        # Lấy Kp hiện tại (để debug xem xe đang dùng Kp đường thẳng hay đường cong)
        # Hack nhẹ vào class controller để lấy Kp (chỉ dùng cho debug)
        abs_alpha = abs(heading)
        if abs_alpha <= controller.alpha_straight: kp_debug = controller.Kp_straight
        elif abs_alpha >= controller.alpha_curve: kp_debug = controller.Kp_curve
        else: kp_debug = float(controller.gain_spline(abs_alpha))

        # --- BƯỚC 3: ACTUATION (Mô phỏng logic LaneWorker) ---
        
        # Logic góc lái: Kẹp [-25, 25]
        steer_final = int(np.clip(steer_deg_raw, -25, 25))
        
        # Logic tốc độ: Nhân 10 và kẹp [-500, 500]
        speed_scaled = target_speed * 10
        speed_final = int(np.clip(speed_scaled, -500, 500))

        # --- BƯỚC 4: LOGGING & VISUALIZATION ---
        
        # Ghi log
        csv_writer.writerow([
            frame_count, f"{current_time:.3f}",
            f"{offset:.4f}", f"{heading:.4f}", f"{np.rad2deg(heading):.2f}", f"{curvature:.0f}",
            f"{target_speed:.2f}", f"{kp_debug:.2f}",
            steer_final, speed_final
        ])

        # Vẽ lên hình
        debug_frame = draw_dashboard(
            frame, steer_final, speed_final, 
            offset, heading, curvature, target_speed
        )
        
        out.write(debug_frame)
        
        # Hiển thị (Optional)
        cv2.imshow('Debug View', debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Hoàn tất! Hãy kiểm tra file debug_output.mp4 và debug_data.csv")

if __name__ == "__main__":
    main()