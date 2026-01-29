import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import utils 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt 

history_steering = []
history_speed = []


# --- CẤU HÌNH CỐ ĐỊNH ---
W, H = 480, 240
widthTop = 102
heightTop = 114
widthBottom = 30
heightBottom = 214
LANE_WIDTH = 260    
MARGIN = 60         
MINPIX = 40         
SMOOTH_KERNEL = 25  

class AdaptiveController:
    def __init__(self):
        # Cấu hình tốc độ (Tham khảo controllerSP.py)
        self.max_speed = 50    # Chạy thẳng
        self.min_speed = 20    # Vào cua
        
        # Ngưỡng góc lái để quyết định tốc độ (rad)
        # 1 độ ~ 0.017 rad, 11 độ ~ 0.19 rad
        self.alpha_straight = np.deg2rad(2)  
        self.alpha_curve = np.deg2rad(15)

        # Cấu hình độ nhạy lái (Gain)
        self.Kp_straight = 0.6  # Đi thẳng lái nhẹ thôi cho đỡ lắc
        self.Kp_curve = 1.5     # Vào cua lái gắt hơn để bám đường

        # --- TẠO BỘ NỘI SUY (CUBIC SPLINE) ---
        self.speed_spline = CubicSpline(
            [self.alpha_straight, self.alpha_curve], 
            [self.max_speed, self.min_speed],
            bc_type=((1, 0.0), (1, 0.0)) # Đạo hàm tại biên bằng 0 (làm phẳng đầu cuối)
        )
        
        self.gain_spline = CubicSpline(
            [self.alpha_straight, self.alpha_curve],
            [self.Kp_straight, self.Kp_curve],
            bc_type=((1, 0.0), (1, 0.0))
        )

    def get_control(self, offset, heading_error):
        # Lấy giá trị tuyệt đối của góc lệch hướng
        abs_alpha = abs(heading_error)

        # 1. Tính toán Tốc độ và Gain dựa trên độ gắt của khúc cua
        if abs_alpha < self.alpha_straight:
            target_speed = self.max_speed
            kp = self.Kp_straight
        elif abs_alpha > self.alpha_curve:
            target_speed = self.min_speed
            kp = self.Kp_curve
        else:
            # Nội suy mượt mà ở khoảng giữa
            target_speed = float(self.speed_spline(abs_alpha))
            kp = float(self.gain_spline(abs_alpha))

        # 2. Tính góc lái (Kết hợp offset và heading)
        # Công thức Stanley Controller đơn giản hóa:
        # Steering = Heading_Error + arctan(k * CrossTrack_Error / speed)
        # Ở đây ta dùng phiên bản đơn giản hơn lai PID:
        
        steering_angle = -kp * offset - 0.8 * heading_error 
        
        # Clip góc lái (-25 đến 25 độ)
        steering_angle_deg = np.rad2deg(steering_angle)
        steering_angle_deg = np.clip(steering_angle_deg, -25, 25)

        return steering_angle_deg, target_speed
    
# Khởi tạo Controller
controller = AdaptiveController()

def get_curve_points(imgWarp):
    h, w = imgWarp.shape[:2]
    ignore_border = 50
    border_drop = 40
    
    # 1. Histogram
    region_ratio = 0.75
    y0 = int(h * region_ratio)
    histogram = np.sum(imgWarp[y0:, :], axis=0).astype(np.float32)

    if SMOOTH_KERNEL > 1:
        k = np.ones(SMOOTH_KERNEL, dtype=np.float32) / SMOOTH_KERNEL
        histogram = np.convolve(histogram, k, mode="same")

    if ignore_border > 0:
        histogram[:ignore_border] = 0
        histogram[w - ignore_border:] = 0

    # 2. Base Points
    l0 = int(w * 0.05); l1 = int(w * 0.45)
    r0 = int(w * 0.55); r1 = int(w * 0.95)
    l0, l1 = max(0, l0), min(w, l1)
    r0, r1 = max(0, r0), min(w, r1)
    
    if l1 <= l0 or r1 <= r0: return None, None, None, None, None

    leftx_base = int(np.argmax(histogram[l0:l1]) + l0)
    rightx_base = int(np.argmax(histogram[r0:r1]) + r0)

    # 3. Nonzero
    nonzero = imgWarp.nonzero()
    nonzeroy = np.array(nonzero[0], dtype=np.int32)
    nonzerox = np.array(nonzero[1], dtype=np.int32)

    if border_drop > 0:
        keep = (nonzerox > border_drop) & (nonzerox < (w - border_drop))
        nonzerox = nonzerox[keep]
        nonzeroy = nonzeroy[keep]

    if nonzerox.size < 200: return None, None, None, None, None

    # 4. Sliding Windows
    nwindows = 12
    window_height = int(h // nwindows)
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if good_left_inds.size > MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if good_right_inds.size > MINPIX:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=np.int32)
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=np.int32)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    y_bottom = h - 1
    y_top = int(h * 0)
    ploty = np.linspace(y_top, y_bottom, y_bottom - y_top + 1)

    # 6. Sanity check
    edge_margin = 60
    if rightx.size >= 200 and np.mean(rightx) > (w - edge_margin):
        return None, None, None, None, None
    if leftx.size >= 200 and np.mean(leftx) < edge_margin:
        return None, None, None, None, None

    # 7. Fallback Logic (Sửa lại trả về 5 giá trị)
    # Left missing, infer from Right
    if leftx.size < 200 and rightx.size >= 200:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        curvature_sign = np.sign(right_fit[0])
        left_fitx = right_fitx - LANE_WIDTH * (1 + 0.15 * curvature_sign)
        # Giả lập left_fit từ right_fit (chỉ thay đổi hệ số C - vị trí)
        left_fit = np.array([right_fit[0], right_fit[1], right_fit[2] - LANE_WIDTH])
        return np.clip(left_fitx, 0, w-1), np.clip(right_fitx, 0, w-1), ploty, left_fit, right_fit

    # Right missing, infer from Left
    if rightx.size < 200 and leftx.size >= 200:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = left_fitx + LANE_WIDTH
        # Giả lập right_fit
        right_fit = np.array([left_fit[0], left_fit[1], left_fit[2] + LANE_WIDTH])
        return np.clip(left_fitx, 0, w-1), np.clip(right_fitx, 0, w-1), ploty, left_fit, right_fit

    if leftx.size < 200 or rightx.size < 200:
        return None, None, None, None, None

    # 8. Fit 2 lane bình thường
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return np.clip(left_fitx, 0, w-1), np.clip(right_fitx, 0, w-1), ploty, left_fit, right_fit

def calculate_control_info(img_shape, left_fit, right_fit):
    """
    Trả về: offset (m), curvature (m), heading_error (rad)
    """
    h, w = img_shape[:2]
    # Tỷ lệ pixel -> mét (Cần đo đạc thực tế để chính xác nhất)
    ym_per_pix = 30 / 240 
    xm_per_pix = 3.7 / 260 
    y_eval = h - 1

    if left_fit is None or right_fit is None:
        return 0, 0, 0
    
    # 1. Tính Offset (Độ lệch tâm)
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    car_center = w / 2
    offset_meter = (lane_center - car_center) * xm_per_pix

    # 2. Tính Curvature (Bán kính cong)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    curvature_meter = (left_curverad + right_curverad) / 2

    # 3. Tính Heading Error (Góc lệch hướng) - MỚI!
    # Đạo hàm f'(y) = 2Ay + B chính là độ dốc (slope) dx/dy
    # Vì trục y hướng xuống, và x hướng ngang, ta tính góc so với trục dọc
    left_slope = 2 * left_fit[0] * y_eval + left_fit[1]
    right_slope = 2 * right_fit[0] * y_eval + right_fit[1]
    avg_slope = (left_slope + right_slope) / 2
    
    # Góc lệch (rad) = arctan(slope)
    # Lưu ý: Đây là góc trong không gian pixel. 
    # Nếu muốn chính xác tuyệt đối cần nhân tỷ lệ xm_per_pix/ym_per_pix, 
    # nhưng ở đây ta dùng xấp xỉ pixel vì Controller tự thích nghi được.
    heading_error_rad = np.arctan(avg_slope)

    return offset_meter, curvature_meter, heading_error_rad

def draw_green_lane(img_original, img_warp, points):
    # 1. Lấy thông tin làn
    left_fitx, right_fitx, ploty, left_fit, right_fit = get_curve_points(img_warp)
    
    if left_fitx is None: return img_original 

    # 2. Tính toán thông số (Lấy thêm Heading Error)
    offset_meter, curvature_meter, heading_error_rad = calculate_control_info(img_warp.shape, left_fit, right_fit)
    
    # 3. Lấy tín hiệu điều khiển từ Adaptive Controller
    steering_angle, speed = controller.get_control(offset_meter, heading_error_rad)

    history_steering.append(steering_angle)
    history_speed.append(speed)

    # 4. Vẽ vùng xanh
    warp_zero = np.zeros_like(img_warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Vẽ viền màu đỏ/xanh dương cho dễ nhìn
    cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), 15) 
    cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), 15) 

    newwarp = utils.warpImg(color_warp, points, W, H, inv=True) 
    result = cv2.addWeighted(img_original, 1, newwarp, 0.5, 0)

    # 5. Hiển thị thông số đầy đủ
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Cột trái
    cv2.putText(result, f"Offset: {offset_meter:.2f} m", (20, 40), font, 0.7, (255, 255, 0), 2)
    cv2.putText(result, f"Heading: {np.rad2deg(heading_error_rad):.1f} deg", (20, 70), font, 0.7, (255, 255, 0), 2)
    cv2.putText(result, f"Radius: {curvature_meter:.0f} m", (20, 100), font, 0.7, (255, 255, 0), 2)
    
    # Cột phải (Kết quả điều khiển)
    cv2.putText(result, f"STEER: {steering_angle:.2f}", (300, 40), font, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"SPEED: {speed:.0f}", (300, 70), font, 0.7, (0, 255, 0), 2)

    print(f"Angle: {steering_angle:.1f} \t Speed: {speed:.0f} \n")

    return result

def frame_processor(image):
    imgSmall = cv2.resize(image, (W, H))
    imgBGR = cv2.cvtColor(imgSmall, cv2.COLOR_RGB2BGR)
    
    imgThres = utils.thresholding(imgBGR)
    
    kernel = np.ones((3,3), np.uint8)
    imgThres = cv2.medianBlur(imgThres, 5)
    imgThres = cv2.morphologyEx(imgThres, cv2.MORPH_CLOSE, kernel, iterations=2)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)

    points = np.float32([(widthTop, heightTop), (W-widthTop, heightTop),
                         (widthBottom, heightBottom), (W-widthBottom, heightBottom)])
    
    imgWarp = utils.warpImg(imgThres, points, W, H)

    try:
        # Gọi draw_green_lane đã được cập nhật logic tính toán
        final_img = draw_green_lane(imgBGR, imgWarp, points)
    except Exception as e:
        print(f"Error processing frame: {e}")
        final_img = imgBGR

    return cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

# --- CHẠY ---
if __name__ == "__main__":
    input_path = 'lane-detection/input.mp4'
    output_path = 'output_PID_Control.mp4'
    
    print("Đang xử lý video...")
    try:
        input_video = VideoFileClip(input_path, audio=False)
        processed = input_video.fl_image(frame_processor)
        processed.write_videofile(output_path, audio=False)
        print("Hoàn tất!")
    except Exception as e:
        print(f"Lỗi khi mở video: {e}")


    # --- PLOTTING CODE (ADD THIS AT THE END) ---
    print("Plotting analysis data...")
    plt.figure(figsize=(10, 8))

    # Plot Steering
    plt.subplot(2, 1, 1)
    plt.plot(history_steering, color='blue', label='Steering Angle')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title("Steering Angle over Time")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True)

    # Plot Speed
    plt.subplot(2, 1, 2)
    plt.plot(history_speed, color='green', label='Speed')
    plt.title("Speed over Time")
    plt.xlabel("Frame")
    plt.ylabel("Speed (cm/s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('analysis_result.png') # Saves the image
    plt.show() # Shows the window
    print("Graph saved as 'analysis_result.png'")