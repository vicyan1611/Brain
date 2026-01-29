import os
import threading
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
    Brake,
    DistanceReading,
)
from src.processing.lane_detection import LaneCurveEstimator

import base64
import numpy as np
import cv2
import time
from queue import Queue, Full, Empty
import torch
import ultralytics
from scipy.interpolate import CubicSpline

class AdaptiveController:
    """
    Điều khiển thích nghi dùng Cubic Spline.
    Input: Offset, Heading.
    Output: Steering Angle (deg), Speed (unit).
    """
    def __init__(self):
        # Cấu hình
        self.max_speed = 50     # Chạy thẳng
        self.min_speed = 20     # Vào cua
        
        self.alpha_straight = np.deg2rad(2)   # Ngưỡng đường thẳng (rad)
        self.alpha_curve = np.deg2rad(15)     # Ngưỡng cua gắt (rad)

        self.Kp_straight = 0.6
        self.Kp_curve = 1.5

        # Cubic Spline Interpolation
        self.speed_spline = CubicSpline(
            [self.alpha_straight, self.alpha_curve], 
            [self.max_speed, self.min_speed],
            bc_type=((1, 0.0), (1, 0.0))
        )
        
        self.gain_spline = CubicSpline(
            [self.alpha_straight, self.alpha_curve],
            [self.Kp_straight, self.Kp_curve],
            bc_type=((1, 0.0), (1, 0.0))
        )

    def get_control(self, offset, heading_error):
        abs_alpha = abs(heading_error)

        # 1. Tính toán Speed & Gain
        if abs_alpha <= self.alpha_straight:
            target_speed = self.max_speed
            kp = self.Kp_straight
        elif abs_alpha >= self.alpha_curve:
            target_speed = self.min_speed
            kp = self.Kp_curve
        else:
            target_speed = float(self.speed_spline(abs_alpha))
            kp = float(self.gain_spline(abs_alpha))

        # 2. Tính Steering (Stanley-like PD)
        # steering = -Kp * offset - Kd * heading
        steering_angle_rad = -kp * offset - 0.8 * heading_error
        
        # Đổi ra độ và giới hạn
        steering_angle_deg = np.rad2deg(steering_angle_rad)
        steering_angle_deg = np.clip(steering_angle_deg, -25, 25)

        return steering_angle_deg, target_speed

class FrameReader(ThreadWithStop):
    """Reads frames from `serialCamera` messages and pushes decoded frames into a local queue."""

    def __init__(self, queuesList, frame_queue, logger=None, pause=0.01, distance_threshold_cm=150.0, log_interval_sec=1.0):
        super(FrameReader, self).__init__(pause=pause)
        self.sub = messageHandlerSubscriber(queuesList, serialCamera, "lastOnly", True)
        self.distance_sub = messageHandlerSubscriber(queuesList, DistanceReading, "lastOnly", True)
        self.q = frame_queue
        self.logger = logger
        self.distance_threshold_cm = distance_threshold_cm
        self.log_interval_sec = log_interval_sec
        self._last_distance_cm = None
        self._last_log_ts = 0.0

    def thread_work(self):
        # Update latest distance if a reading is available
        latest_distance = self.distance_sub.receive()
        if latest_distance is not None:
            self._last_distance_cm = latest_distance
            now = time.time()
            if self.logger and (now - self._last_log_ts) >= self.log_interval_sec:
                within_gate = self._last_distance_cm <= self.distance_threshold_cm
                self.logger.info(
                    "DistanceReader: %.2f cm (gate=%s, thr=%.0f)",
                    self._last_distance_cm,
                    within_gate,
                    self.distance_threshold_cm,
                )
                self._last_log_ts = now

        msg = self.sub.receive()
        if msg is None:
            return

        # Drop frames if we are too far from the target
        if self._last_distance_cm is not None and self._last_distance_cm > self.distance_threshold_cm:
            return
        try:
            # expect base64-encoded jpeg string
            data = base64.b64decode(msg)
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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
        self.brake_sender = messageHandlerSender(self.queuesList, Brake)   # optional
        self._last_stop_time = 0

    def thread_work(self):
        try:
            frame = self.q.get(timeout=0.5)
        except Empty:
            return

        try:
            h, w = frame.shape[:2]
            cx1 = int(w * 0.3)
            cy1 = int(h * 0.3)
            cx2 = int(w * 0.7)
            cy2 = int(h * 0.7)
            crop = frame[cy1:cy2, cx1:cx2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(edges.mean() / 255.0)
            if self.logger:
                self.logger.info("Perception obstacle edge_density=%.4f", edge_density)

            # threshold and simple rate-limit
            if edge_density > 0.06:
                now = time.time()
                if now - self._last_stop_time > 1.0:
                    self._last_stop_time = now
                    try:
                        self.speed_sender.send("0")
                        self.brake_sender.send("0")
                    except Exception:
                        pass
                    try:
                        self.warn_sender.send(f"obstacle:{edge_density:.4f}")
                    except Exception:
                        pass
        except Exception as e:
            if self.logger:
                self.logger.debug("ObstacleWorker error: %s", e)

class ObstacleWorkerYOLO(BasePerceptionWorker):
    """Detect obstacles using YOLOv8 model."""
    def __init__(self, frame_queue, queuesList, logger=None, pause=0.01,
                yolo_model=None, device="cpu", lock=None, score_thr=0.35, save_dir=None):
        
        super(ObstacleWorkerYOLO, self).__init__(frame_queue, queuesList, logger, pause)
        self.speed_sender = messageHandlerSender(queuesList, SpeedMotor)
        self.warn_sender = messageHandlerSender(queuesList, WarningSignal)
        self.brake_sender = messageHandlerSender(self.queuesList, Brake)
        self.model = yolo_model
        self.device = device
        self.lock = lock or threading.Lock()
        self.score_thr = score_thr
        self.save_dir = save_dir
        self._frame_idx = 0
        self._last_stop_time = 0

    def thread_work(self):
        try:
            frame = self.q.get(timeout=0.5)
        except Empty:
            return

        try:
            with torch.inference_mode():
                # lock to avoid concurrent model() calls from multiple threads
                with self.lock:
                    results = self.model(frame, verbose=False, device=self.device)[0]
                    self.logger.debug("ObstacleWorkerYOLO: %d boxes detected", len(results.boxes))

            for box in results.boxes:
                conf = float(box.conf)
                if conf < self.score_thr:
                    continue
                cls_id = int(box.cls)
                xyxy = box.xyxy[0].tolist()

                # ----- Handle detected obstacle -----
                if self.save_dir:
                    self._frame_idx += 1
                    vis = frame.copy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{cls_id}:{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    out_path = os.path.join(self.save_dir, f"frame_{self._frame_idx:06d}.jpg")
                    cv2.imwrite(out_path, vis)                    
                # ----- Finish handling obstacle -----

                # # simple action: stop and warn once per second
                # now = time.time()
                # if now - self._last_stop_time > 1.0:
                #     self._last_stop_time = now
                #     try:
                #         self.speed_sender.send("0")
                #         self.brake_sender.send("0")
                #     except Exception:
                #         pass
                #     try:
                #         self.warn_sender.send(f"yolo:{cls_id}:{conf:.2f}")
                #     except Exception:
                #         pass
        except Exception as e:
            if self.logger:
                self.logger.debug("ObstacleWorkerYOLO error: %s", e)

        

class LaneWorker(BasePerceptionWorker):
    """
    Lane detection worker that uses Adaptive Controller.
    Controls BOTH Steer and Speed based on road curvature.
    """
    def __init__(self, frame_queue, queuesList, logger=None, pause=0.02):
        super(LaneWorker, self).__init__(frame_queue, queuesList, logger, pause)
        
        # Senders
        self.steer_sender = messageHandlerSender(queuesList, SteerMotor)
        self.speed_sender = messageHandlerSender(queuesList, SpeedMotor) 
        self.lane_sender = messageHandlerSender(queuesList, LaneKeeping)
        
        # Khởi tạo Logic
        self.estimator = LaneCurveEstimator()
        self.controller = AdaptiveController() 

    def thread_work(self):
        try:
            frame = self.q.get(timeout=0.5)
        except Empty:
            return
        try:
            # 1. Perception: Lấy thông số từ ảnh
            offset, curvature, heading, _ = self.estimator.process(frame)
            
            # 2. Control: Tính góc lái và tốc độ
            steer_deg, target_speed = self.controller.get_control(offset, heading)
            
            # 3. Actuation: Gửi tín hiệu
            steer_final = float(np.clip(steer_deg, -25, 25))
            self.steer_sender.send(int(steer_final))

            speed_scaled = target_speed * 10
            speed_final = float(np.clip(speed_scaled, -500, 500))
            self.speed_sender.send(int(speed_final))

            # Debug log
            if self.logger:
                self.logger.debug(
                    "Lane: Off=%.2f Head=%.2f | Steer=%.1f Speed=%d", 
                    offset, np.rad2deg(heading), steer_deg, int(target_speed)
                )

        except Exception as e:
            if self.logger:
                self.logger.debug("LaneWorker error: %s", e)


class processPerception(WorkerProcess):
    """Perception process that starts a frame reader and multiple worker threads.

    Add new worker types by creating a ThreadWithStop subclass and appending
    it in `_init_threads`.
    """

    def __init__(self, queueList, logging, ready_event=None, debugging=False, distance_threshold_cm=100.0, distance_log_interval_sec=1.0):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.ready_event = ready_event
        self._frame_queue = Queue(maxsize=4)
        self.distance_threshold_cm = distance_threshold_cm
        self.distance_log_interval_sec = distance_log_interval_sec
        super(processPerception, self).__init__(self.queuesList, ready_event)
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _init_model(self):
        self.model_path = 'models/yolov8n.pt'
        self.model = ultralytics.YOLO(self.model_path)
        self.model.to(self.device)
        self.model.fuse()
        self.model_lock = threading.Lock()
        self.logging.info("Perception YOLO model loaded on %s", self.device)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("detected_objects", ts)
        os.makedirs(self.save_dir, exist_ok=True)
        self.logging.info("Perception YOLO model loaded on %s", self.device)

    def _init_threads(self):
        # Frame reader
        self.threads.append(
            FrameReader(
                self.queuesList,
                self._frame_queue,
                self.logging,
                distance_threshold_cm=self.distance_threshold_cm,
                log_interval_sec=self.distance_log_interval_sec,
            )
        )

        # Worker threads (easy to extend)
        # self.threads.append(ObstacleWorker(self._frame_queue, self.queuesList, self.logging))

        # self.threads.append(
        #     ObstacleWorkerYOLO(
        #         self._frame_queue,
        #         self.queuesList,
        #         self.logging,
        #         yolo_model=self.model,
        #         device=self.device,
        #         lock=self.model_lock,
        #         score_thr=0.35,
        #         save_dir=self.save_dir,
        #     )
        # )
        
        self.threads.append(LaneWorker(self._frame_queue, self.queuesList, self.logging))

        # Add more workers here as needed

    def state_change_handler(self):
        # no process-wide state handling for now
        pass

    def process_work(self):
        # nothing here; workers run independently
        pass
