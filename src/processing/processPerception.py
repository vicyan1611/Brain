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
    """Simple lane detection placeholder; computes steering angle and sends it."""

    def __init__(self, frame_queue, queuesList, logger=None, pause=0.02):
        super(LaneWorker, self).__init__(frame_queue, queuesList, logger, pause)
        self.steer_sender = messageHandlerSender(queuesList, SteerMotor)
        self.lane_sender = messageHandlerSender(queuesList, LaneKeeping)

    def thread_work(self):
        try:
            frame = self.q.get(timeout=0.5)
        except Empty:
            return

        try:
            # Placeholder lane algorithm: compute center of bright region as lane center
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(thr)
            h, w = frame.shape[:2]
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                offset = (cx - w // 2)
                # simple proportional steering (tune in real system)
                steering_angle = -offset * 0.1
                # send steer and lane offset
                try:
                    self.steer_sender.send(str(int(steering_angle)))
                except Exception:
                    pass
                try:
                    self.lane_sender.send(int(offset))
                except Exception:
                    pass
            else:
                # no lane detected
                pass
        except Exception as e:
            if self.logger:
                self.logger.debug("LaneWorker error: %s", e)


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
