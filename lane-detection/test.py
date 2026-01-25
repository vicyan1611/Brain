import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import utils 

# --- CẤU HÌNH CỐ ĐỊNH (SIZE 480x240) ---
# Kích thước chuẩn
W, H = 480, 240

# Thông số Trackbars (đã chốt)
widthTop = 102
heightTop = 114
widthBottom = 30
heightBottom = 214

# Thông số thuật toán (đã chốt cho size nhỏ)
LANE_WIDTH = 260    
MARGIN = 60         
MINPIX = 40         
SMOOTH_KERNEL = 25  

def get_curve_points(imgWarp):
    """
    Tìm lane trên ảnh Warp kích thước 480x240
    """
    h, w = imgWarp.shape[:2]

    ignore_border = 50
    border_drop = 40
    
    # -------------------------
    # 1) Histogram & Smoothing
    # -------------------------
    region_ratio = 0.75
    y0 = int(h * region_ratio)
    histogram = np.sum(imgWarp[y0:, :], axis=0).astype(np.float32)

    if SMOOTH_KERNEL > 1:
        k = np.ones(SMOOTH_KERNEL, dtype=np.float32) / SMOOTH_KERNEL
        histogram = np.convolve(histogram, k, mode="same")

    if ignore_border > 0:
        histogram[:ignore_border] = 0
        histogram[w - ignore_border:] = 0

    # -------------------------
    # 2) Tìm điểm khởi đầu (Base)
    # -------------------------
    l0 = int(w * 0.05); l1 = int(w * 0.45)
    r0 = int(w * 0.55); r1 = int(w * 0.95)

    l0, l1 = max(0, l0), min(w, l1)
    r0, r1 = max(0, r0), min(w, r1)
    
    if l1 <= l0 or r1 <= r0: return None, None, None

    leftx_base = int(np.argmax(histogram[l0:l1]) + l0)
    rightx_base = int(np.argmax(histogram[r0:r1]) + r0)

    # -------------------------
    # 3) Lấy tất cả điểm ảnh (Nonzero pixels)
    # -------------------------
    nonzero = imgWarp.nonzero()
    nonzeroy = np.array(nonzero[0], dtype=np.int32)
    nonzerox = np.array(nonzero[1], dtype=np.int32)

    if border_drop > 0:
        keep = (nonzerox > border_drop) & (nonzerox < (w - border_drop))
        nonzerox = nonzerox[keep]
        nonzeroy = nonzeroy[keep]

    if nonzerox.size < 200: return None, None, None

    # -------------------------
    # 4) Sliding windows
    # -------------------------
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

    # ploty = np.linspace(0, h - 1, h) # này nhìn tới đường chân trời
    y_bottom = h - 1
    y_top = int(h * 0)   # chỉ nhìn 100% phía trước, sau này có thể chỉnh nha Phúc

    ploty = np.linspace(y_top, y_bottom, y_bottom - y_top + 1)



    # -------------------------
    # 6) Sanity check (Kiểm tra bám biên)
    # -------------------------
    edge_margin = 60
    if rightx.size >= 200 and np.mean(rightx) > (w - edge_margin):
        return None, None, None
    if leftx.size >= 200 and np.mean(leftx) < edge_margin:
        return None, None, None

    # -------------------------
    # 7) Fallback 1 lane (Suy luận làn còn thiếu)
    # -------------------------
    # Chỉ có Right Lane -> Suy ra Left Lane
    if leftx.size < 200 and rightx.size >= 200:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # Logic bẻ cong làn suy luận
        curvature_sign = np.sign(right_fit[0])
        left_fitx = right_fitx - LANE_WIDTH * (1 + 0.15 * curvature_sign)
        
        return np.clip(left_fitx, 0, w-1), np.clip(right_fitx, 0, w-1), ploty

    # Chỉ có Left Lane -> Suy ra Right Lane
    if rightx.size < 200 and leftx.size >= 200:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = left_fitx + LANE_WIDTH 
        return np.clip(left_fitx, 0, w-1), np.clip(right_fitx, 0, w-1), ploty

    if leftx.size < 200 or rightx.size < 200:
        return None, None, None

    # -------------------------
    # 8) Fit 2 lane bình thường
    # -------------------------
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return np.clip(left_fitx, 0, w-1), np.clip(right_fitx, 0, w-1), ploty

def draw_green_lane(img_original, img_warp, points):
    """ Vẽ vùng xanh lên ảnh """
    left_fitx, right_fitx, ploty = get_curve_points(img_warp)
    
    if left_fitx is None: return img_original 

    warp_zero = np.zeros_like(img_warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Vẽ màu xanh lá cây
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Vẽ viền
    cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), 10)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), 10)

    # Warp ngược (Un-warp)
    newwarp = utils.warpImg(color_warp, points, W, H, inv=True) 
    
    # Gộp 2 ảnh
    result = cv2.addWeighted(img_original, 1, newwarp, 0.5, 0)
    return result

def frame_processor(image):
    # Mọi ảnh đi vào đều bị ép về 480x240
    imgSmall = cv2.resize(image, (W, H))
    
    # 2. Xử lý màu (MoviePy dùng RGB -> OpenCV dùng BGR)
    imgBGR = cv2.cvtColor(imgSmall, cv2.COLOR_RGB2BGR)

    # 3. Thresholding
    imgThres = utils.thresholding(imgBGR)

    # 4. Lọc nhiễu
    kernel = np.ones((3,3), np.uint8)
    imgThres = cv2.medianBlur(imgThres, 5)
    imgThres = cv2.morphologyEx(imgThres, cv2.MORPH_CLOSE, kernel, iterations=2)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)

    # 5. Warp (Dùng điểm cố định)
    points = np.float32([(widthTop, heightTop), (W-widthTop, heightTop),
                         (widthBottom, heightBottom), (W-widthBottom, heightBottom)])
    
    imgWarp = utils.warpImg(imgThres, points, W, H)

    # 6. Vẽ Lane (Không cần scale nữa)
    try:
        final_img = draw_green_lane(imgBGR, imgWarp, points)
    except Exception as e:
        print(f"Error: {e}")
        final_img = imgBGR

    # 7. Trả về ảnh 480x240 (RGB)
    return cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

# --- CHẠY ---
input_path = 'input.mp4'
output_path = 'output_480x240_Final.mp4'

print("Đang xử lý video về size chuẩn 480x240...")
input_video = VideoFileClip(input_path, audio=False)
processed = input_video.fl_image(frame_processor)
processed.write_videofile(output_path, audio=False)