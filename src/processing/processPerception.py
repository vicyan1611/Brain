from src.templates.workerprocess import WorkerProcess
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import (
    serialCamera,
    SpeedMotor,
    SteerMotor,
    WarningSignal,
    LaneKeeping,
)

import base64
import numpy as np
import cv2
import time
from queue import Queue, Full, Empty


class FrameReader(ThreadWithStop):
    """Reads frames from `serialCamera` messages and pushes decoded frames into a local queue."""

    def __init__(self, queuesList, frame_queue, logger=None, pause=0.01):
        super(FrameReader, self).__init__(pause=pause)
        self.sub = messageHandlerSubscriber(queuesList, serialCamera, "lastOnly", True)
        self.q = frame_queue
        self.logger = logger

    def thread_work(self):
        msg = self.sub.receive()
        if msg is None:
            return
        try:
            # expect base64-encoded jpeg string
            data = base64.b64decode(msg)
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            try:
                self.q.put_nowait(frame)
            except Full:
                # drop frame if workers are busy
                pass
        except Exception as e:
            if self.logger:
                self.logger.debug("FrameReader decode error: %s", e)


class BasePerceptionWorker(ThreadWithStop):
    """Base class for perception workers; consume frames from shared queue."""

    def __init__(self, frame_queue, queuesList, logger=None, pause=0.01):
        super(BasePerceptionWorker, self).__init__(pause=pause)
        self.q = frame_queue
        self.queuesList = queuesList
        self.logger = logger

    def thread_work(self):
        # override in subclass
        pass


class ObstacleWorker(BasePerceptionWorker):
    """Detect obstacles using simple edge-density on center crop."""

    def __init__(self, frame_queue, queuesList, logger=None, pause=0.01):
        super(ObstacleWorker, self).__init__(frame_queue, queuesList, logger, pause)
        self.speed_sender = messageHandlerSender(queuesList, SpeedMotor)
        self.warn_sender = messageHandlerSender(queuesList, WarningSignal)
        self._last_stop_time = 0

        self._obs_count = 0
        self._require_consecutive = 3   # tune: 2-6 tùy độ rung/nhiễu

    def detect_obstacle(
        self,
        frame,
        roi_y_start=0.55,          # nhìn phần dưới ảnh (55% -> 100%)
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
    
    def thread_work(self):
        try:
            frame = self.q.get(timeout=0.5)
        except Empty:
            return

        # Guard an toàn
        if frame is None or getattr(frame, "size", 0) == 0:
            return

        try:
            # === GỌI HÀM detect_obstacle MỚI (CÓ self) ===
            is_obstacle, score, debug = self.detect_obstacle(
                frame,
                roi_y_start=0.25,          # nhìn phía trước xe (phần dưới ảnh)
                roi_x_left=0.2,
                roi_x_right=0.8,
                blur_ksize=5,
                canny1=60,
                canny2=160,
                edge_ratio_threshold=0.06,
                min_contour_area=800
            )

            if self.logger:
                self.logger.info(
                    "Obstacle detect | is=%s | score=%.4f | edge=%.4f | area=%.1f",
                    is_obstacle,
                    score,
                    float(debug.get("edge_ratio", 0.0)),
                    float(debug.get("max_contour_area", 0.0)),
                )

            # === DEBOUNCE: phải gặp liên tiếp N frame ===
            if is_obstacle:
                self._obs_count += 1
            else:
                self._obs_count = 0

            if self._obs_count >= self._require_consecutive:
                now = time.time()

                # === RATE LIMIT: tránh spam stop ===
                if now - self._last_stop_time > 1.0:
                    self._last_stop_time = now

                    # === STOP XE ===
                    try:
                        self.speed_sender.send("0")
                    except Exception:
                        pass

                    # === CẢNH BÁO ===
                    try:
                        self.warn_sender.send(
                            f"obstacle score={score:.4f} "
                            f"edge={debug.get('edge_ratio', 0.0):.4f} "
                            f"area={debug.get('max_contour_area', 0.0):.1f}"
                        )
                    except Exception:
                        pass

        except Exception as e:
            if self.logger:
                self.logger.debug("ObstacleWorker error: %s", e)


class LaneWorker(BasePerceptionWorker):
    """
    Lane Detection using Histogram & Dynamic Thresholding (Ported from C++ Team Code).
    """

    def __init__(self, frame_queue, queuesList, logger=None, pause=0.02):
        super(LaneWorker, self).__init__(frame_queue, queuesList, logger, pause)
        self.steer_sender = messageHandlerSender(queuesList, SteerMotor)
        self.lane_sender = messageHandlerSender(queuesList, LaneKeeping)
        self.speed_sender = messageHandlerSender(queuesList, SpeedMotor) # Cần để dừng khi gặp Stopline

        # Tuning Parameters
        self.kp = 0.15          # Hệ số đánh lái (Tune: 0.1 -> 0.5)
        self.max_angle = 25     # Góc lái tối đa
        self.prev_center = 320  # Giả sử tâm ảnh là 320 (với ảnh 640x480)
        
        # Logic stopline
        self.stopline_detected = False

    def extract_lanes(self, hist_data):
        """
        Tìm các chỉ số cột (index) nơi bắt đầu hoặc kết thúc vạch kẻ đường.
        Logic: Chuyển từ 0 lên cao (edge lên) hoặc từ cao xuống 0 (edge xuống).
        """
        lane_indices = []
        previous_value = 0
        
        # hist_data là mảng 1 chiều (640 phần tử)
        for idx, value in enumerate(hist_data):
            # Threshold 1500 tương đương khoảng 6 pixel trắng (255*6 ~ 1530)
            if value >= 1500 and previous_value == 0:
                lane_indices.append(idx)
                previous_value = 255
            elif value == 0 and previous_value == 255:
                lane_indices.append(idx)
                previous_value = 0
                
        # Nếu số lượng điểm lẻ, thêm điểm cuối cùng của ảnh vào
        if len(lane_indices) % 2 == 1:
            lane_indices.append(len(hist_data) - 1)
            
        return lane_indices

    def process_histogram_algorithm(self, frame):
        """
        Logic chính port từ hàm optimized_histogram của C++
        """
        h, w = frame.shape[:2]
        self.stopline_detected = False
        
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. ROI Selection (Bottom part of image like C++ code: y=384, h=96)
        # Lưu ý: C++ dùng cv::Rect(0, 384, 640, 96)
        roi_y_start = 384
        if roi_y_start >= h: roi_y_start = h - 100 # Fallback nếu ảnh nhỏ hơn
        roi = gray[roi_y_start:h, 0:w]
        
        # 3. Dynamic Thresholding (Key Feature!)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(roi)
        
        # Logic C++: double threshold_value = std::min(std::max(maxVal - 55.0, 30.0), 200.0);
        threshold_value = max(maxVal - 55.0, 30.0)
        threshold_value = min(threshold_value, 200.0)
        
        _, thresh = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 4. Histogram Calculation (Reduce sum columns)
        # Axis 0 là cộng dồn theo cột dọc
        hist = np.sum(thresh, axis=0) 
        
        # 5. Extract Lanes
        lanes = self.extract_lanes(hist)
        centers = []
        
        # Tính trung điểm từng cặp vạch (lanes[2*i] và lanes[2*i+1])
        # lanes structure: [start_L, end_L, start_R, end_R, ...]
        num_pairs = len(lanes) // 2
        for i in range(num_pairs):
            start = lanes[2 * i]
            end = lanes[2 * i + 1]
            width_lane = abs(start - end)
            
            # Logic C++ Stopline: abs(...) > 350 && thresh > 50
            if width_lane > 350 and threshold_value > 50:
                self.stopline_detected = True
                return w / 2.0, threshold_value # Return center mặc định nếu gặp stopline
            
            # Logic lọc nhiễu: chỉ lấy vạch có độ rộng > 3 pixel
            if width_lane > 3:
                centers.append((start + end) / 2.0)

        # 6. Calculate Final Center based on visible lanes
        target_center = w / 2.0
        
        if not centers:
            # Không thấy đường -> Giữ lái thẳng hoặc giá trị cũ
            target_center = w / 2.0
        elif len(centers) == 1:
            # Chỉ thấy 1 vạch
            c = centers[0]
            if c > (w / 2.0):
                # Thấy vạch phải -> xe đang lệch trái -> Tâm đường nằm bên trái vạch này
                # Logic C++: (centers[0] - 0) / 2 ... Hơi lạ, logic này có thể làm xe bám sát lề
                # Ta sẽ điều chỉnh logic này an toàn hơn: Giả sử đường rộng 300px
                target_center = c - 150 
            else:
                # Thấy vạch trái
                target_center = c + 150
        elif abs(centers[0] - centers[-1]) < 200:
             # Hai vạch quá gần nhau -> Có thể là nhiễu hoặc đường hẹp, lấy trung bình
             avg = (centers[0] + centers[-1]) / 2.0
             if avg > w: 
                 target_center = w/2 # Fallback
             else:
                 target_center = (centers[0] + centers[-1] + w) / 2 if avg > w/2 else (centers[0] + centers[-1])/2
                 # Đoạn logic C++ khúc này hơi rối rắm, ta đơn giản hóa:
                 target_center = (centers[0] + centers[-1]) / 2.0
        else:
            # Trường hợp lý tưởng: Thấy vạch trái ngoài cùng và vạch phải ngoài cùng
            target_center = (centers[0] + centers[-1]) / 2.0

        return target_center, threshold_value

    def thread_work(self):
        try:
            frame = self.q.get(timeout=0.5)
        except Empty:
            return

        if frame is None or getattr(frame, "size", 0) == 0:
            return

        try:
            h, w = frame.shape[:2]
            img_center_x = w // 2
            
            # === CHẠY THUẬT TOÁN HISTOGRAM ===
            lane_center, thresh_val = self.process_histogram_algorithm(frame)
            
            # === XỬ LÝ STOPLINE ===
            if self.stopline_detected:
                # Gửi lệnh dừng xe
                try:
                    self.speed_sender.send("0")
                    self.lane_sender.send("STOPLINE DETECTED")
                except: pass
                # Reset steering về 0
                steering_angle = 0
            else:
                # === TÍNH GÓC LÁI (PID P-Controller) ===
                # Error = Tâm đường mong muốn - Tâm xe (giữa ảnh)
                error = lane_center - img_center_x
                
                # C++ Logic có đoạn previous_center smoothing, ta áp dụng nhẹ
                # Low-pass filter để góc lái mượt hơn
                lane_center = 0.7 * lane_center + 0.3 * self.prev_center
                self.prev_center = lane_center
                
                # Tính lại error sau khi smooth
                error = lane_center - img_center_x
                
                steering_angle = int(error * self.kp)
                
                # Clamp góc lái
                steering_angle = max(-self.max_angle, min(self.max_angle, steering_angle))
                
                # Gửi tín hiệu lái
                try:
                    self.steer_sender.send(str(steering_angle))
                except Exception:
                    pass

            # === DEBUG INFO ===
            try:
                msg = (f"Steer:{steering_angle} | Center:{int(lane_center)} | "
                       f"Thresh:{int(thresh_val)} | Stop:{self.stopline_detected}")
                self.lane_sender.send(msg)
            except Exception:
                pass

        except Exception as e:
            if self.logger:
                self.logger.debug("LaneWorker Hist Error: %s", e)

class processPerception(WorkerProcess):
    """Perception process that starts a frame reader and multiple worker threads.

    Add new worker types by creating a ThreadWithStop subclass and appending
    it in `_init_threads`.
    """

    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self._frame_queue = Queue(maxsize=4)
        super(processPerception, self).__init__(self.queuesList, ready_event)

    def _init_threads(self):
        # Frame reader
        self.threads.append(FrameReader(self.queuesList, self._frame_queue, self.logging))

        # Worker threads (easy to extend)
        self.threads.append(ObstacleWorker(self._frame_queue, self.queuesList, self.logging))
        # self.threads.append(LaneWorker(self._frame_queue, self.queuesList, self.logging)) # Chưa xài nên cmt

        # Add more workers here as needed

    def state_change_handler(self):
        # no process-wide state handling for now
        pass

    def process_work(self):
        # nothing here; workers run independently
        pass
