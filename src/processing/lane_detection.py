import cv2
import numpy as np

# --- CẤU HÌNH CỐ ĐỊNH ---
_BASE_W = 480
_BASE_H = 240
LANE_WIDTH_PX = 260
MARGIN = 60
MINPIX = 40
SMOOTH_KERNEL = 25

# Tỷ lệ thực tế (Cần calibrate lại nếu có thể)
YM_PER_PIX = 30 / 240
XM_PER_PIX = 3.7 / 260

class LaneCurveEstimator:
    """
    Advanced Lane Detection using Sliding Windows & Polynomial Fit.
    Returns: offset (m), curvature (m), heading_error (rad), debug_frame
    """

    def __init__(self):
        widthTop = 102
        heightTop = 114
        widthBottom = 30
        heightBottom = 214
        
        self.src_points = np.float32([
            (widthTop, heightTop), 
            (_BASE_W - widthTop, heightTop),
            (widthBottom, heightBottom), 
            (_BASE_W - widthBottom, heightBottom)
        ])
        
        self.dst_points = np.float32([
            [0, 0], 
            [_BASE_W, 0], 
            [0, _BASE_H], 
            [_BASE_W, _BASE_H]
        ])
        
        # Ma trận biến đổi
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def _thresholding(self, img):
        """Lọc màu trắng/vàng để tách làn"""
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # White mask
        lowerWhite = np.array([0, 0, 200])
        upperWhite = np.array([179, 55, 255])
        maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
        return maskWhite

    def _warp_img(self, img):
        return cv2.warpPerspective(img, self.M, (_BASE_W, _BASE_H))
    
    # def warpImg(img, points, w, h, inv = False):
    #     pts1 = np.float32(points)
    #     pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    #     if inv:
    #         matrix = cv2.getPerspectiveTransform(pts2, pts1)
    #     else:
    #         matrix = cv2.getPerspectiveTransform(pts1, pts2)
    #     imgWarp = cv2.warpPerspective(img, matrix, (w,h))
    #     return imgWarp

    def _find_lane_pixels(self, img_warp):
        h, w = img_warp.shape[:2]
        
        # 1. Histogram tìm điểm khởi đầu
        y0 = int(h * 0.75)
        histogram = np.sum(img_warp[y0:, :], axis=0)
        
        # Smooth histogram
        if SMOOTH_KERNEL > 1:
            # k = np.ones(SMOOTH_KERNEL) / SMOOTH_KERNEL
            k = np.ones(SMOOTH_KERNEL, dtype=np.float32) / SMOOTH_KERNEL
            histogram = np.convolve(histogram, k, mode='same')
            
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # 2. Sliding Windows
        nwindows = 12
        window_height = int(h // nwindows)
        
        nonzero = img_warp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
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
            
            if len(good_left_inds) > MINPIX:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > MINPIX:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def process(self, frame):
        """
        Xử lý frame chính.
        Input: Frame ảnh gốc (BGR)
        Output: (offset, curvature, heading_error, processed_frame)
        """
        frame = cv2.resize(frame, (_BASE_W, _BASE_H))
        
        # Pipeline xử lý ảnh
        img_thresh = self._thresholding(frame)
        img_warp = self._warp_img(img_thresh)
        
        # Tìm pixel làn đường
        leftx, lefty, rightx, righty = self._find_lane_pixels(img_warp)
        
        ploty = np.linspace(0, _BASE_H-1, _BASE_H)
        
        # Fit đa thức & Fallback logic
        left_fit = right_fit = None
        
        # Case 1: Đủ cả 2 làn
        if len(leftx) > MINPIX and len(rightx) > MINPIX:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
        # Case 2: Mất làn trái -> Suy ra từ phải
        elif len(leftx) <= MINPIX and len(rightx) > MINPIX:
            right_fit = np.polyfit(righty, rightx, 2)
            left_fit = np.array([right_fit[0], right_fit[1], right_fit[2] - LANE_WIDTH_PX])
            
        # Case 3: Mất làn phải -> Suy ra từ trái
        elif len(rightx) <= MINPIX and len(leftx) > MINPIX:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.array([left_fit[0], left_fit[1], left_fit[2] + LANE_WIDTH_PX])
            
        # Case 4: Mất cả 2 (Trả về None)
        else:
            return 0.0, 0.0, 0.0, frame

        # --- TÍNH TOÁN METRICS ---
        y_eval = _BASE_H - 1
        
        # 1. Offset
        left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_x_bottom + right_x_bottom) / 2
        car_center = _BASE_W / 2
        offset_meter = (lane_center - car_center) * XM_PER_PIX
        
        # 2. Heading Error (Góc lệch hướng xe)
        left_slope = 2 * left_fit[0] * y_eval + left_fit[1]
        right_slope = 2 * right_fit[0] * y_eval + right_fit[1]
        avg_slope = (left_slope + right_slope) / 2
        heading_error_rad = np.arctan(avg_slope)
        
        # 3. Curvature
        left_curverad = ((1 + (2*left_fit[0]*y_eval*YM_PER_PIX + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*YM_PER_PIX + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        curvature_meter = (left_curverad + right_curverad) / 2

        return offset_meter, curvature_meter, heading_error_rad, frame