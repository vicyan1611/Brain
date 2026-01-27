# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.
#
# Data collection process: records camera frames + telemetry (speed, steer) to
# disk for ML training. Produces per-session video + CSV + ZIP.

import base64
import csv
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.templates.workerprocess import WorkerProcess
from src.utils.messages.allMessages import (
    DataCollectionCommand,
    DataCollectionStatus,
    SpeedMotor,
    SteerMotor,
    mainCamera,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber


class processDataCollector(WorkerProcess): 
    """Process that listens for camera frames + motor commands and writes
    synchronized video and CSV telemetry for training datasets.
    """

    def __init__(self, queueList, logging, ready_event=None, debugging: bool = False):
        self.queuesList = queueList
        self.logger = logging
        self.debugging = debugging

        # Paths
        self.base_path = Path(__file__).resolve().parents[2]  # repo root
        self.data_root = self.base_path / "temp" / "data_collection"
        self.data_root.mkdir(parents=True, exist_ok=True)

        # State
        self.recording = False
        self.session_name: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.video_path: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.zip_path: Optional[Path] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_count = 0
        self.start_time = 0.0
        self.target_size: Optional[Tuple[int, int]] = None
        self.last_speed = 0.0
        self.last_steer = 0.0

        # IPC
        self.commandSubscriber = messageHandlerSubscriber(
            self.queuesList, DataCollectionCommand, "lastOnly", True
        )
        self.cameraSubscriber = messageHandlerSubscriber(
            self.queuesList, mainCamera, "lastOnly", True
        )
        self.speedSubscriber = messageHandlerSubscriber(
            self.queuesList, SpeedMotor, "lastOnly", True
        )
        self.steerSubscriber = messageHandlerSubscriber(
            self.queuesList, SteerMotor, "lastOnly", True
        )
        self.statusSender = messageHandlerSender(self.queuesList, DataCollectionStatus)

        super(processDataCollector, self).__init__(self.queuesList, ready_event)

    # WorkerProcess contract -------------------------------------------------
    def _init_threads(self):
        """No threads needed; everything happens in process_work."""
        self.threads = []

    def process_work(self):
        # Update latest speed/steer even when idle
        speed = self.speedSubscriber.receive()
        if speed is not None:
            try:
                self.last_speed = float(speed) / 10.0
            except Exception:
                self.last_speed = 0.0

        steer = self.steerSubscriber.receive()
        if steer is not None:
            try:
                self.last_steer = float(steer) / 10.0
            except Exception:
                self.last_steer = 0.0

        # Handle commands
        command = self.commandSubscriber.receive()
        if command is not None:
            self._handle_command(command)

        # Handle frames only when recording
        if self.recording:
            frame_b64 = self.cameraSubscriber.receive()
            if frame_b64 is not None:
                self._write_frame(frame_b64)

        # small sleep is handled by WorkerProcess loop (_blocker.wait)

    # Command handling ------------------------------------------------------
    def _handle_command(self, command):
        action = str(command.get("Action", "")).lower()
        meta = command.get("Meta", {})

        if action == "start":
            self._start_recording(meta)
        elif action == "stop":
            self._stop_recording()
        else:
            self._send_status("error", {"message": f"Unknown action: {action}"})

    # Recording lifecycle ---------------------------------------------------
    def _start_recording(self, meta: dict):
        if self.recording:
            return

        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.session_name = f"data_{timestamp}"
        self.session_dir = self.data_root / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.session_dir / "video.mp4"
        self.csv_path = self.session_dir / "data.csv"
        self.zip_path = None

        # CSV setup
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["timestamp", "frame_id", "speed_motor", "steer_motor"],
        )
        self.csv_writer.writeheader()

        # Video writer will be initialized lazily once the first frame arrives
        self.video_writer = None
        self.target_size = None

        self.frame_count = 0
        self.start_time = time.time()
        self.recording = True
        self._send_status(
            "recording",
            {
                "session": self.session_name,
                "started_at": self.start_time,
                "meta": meta,
            },
        )

    def _stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self._close_outputs()
        self._create_zip()
        self._send_status(
            "idle",
            {
                "session": self.session_name,
                "frames": self.frame_count,
                "zip_path": str(self.zip_path) if self.zip_path else None,
            },
        )

    # Frame writing ---------------------------------------------------------
    def _write_frame(self, frame_b64: str):
        try:
            frame = self._decode_frame(frame_b64)
            if frame is None:
                return

            if self.target_size is None:
                # Downscale to keep file size manageable while preserving aspect ratio
                h, w = frame.shape[:2]
                target_w = 960
                scale = target_w / float(w)
                target_h = int(h * scale)
                self.target_size = (target_w, target_h)

            if self.target_size:
                frame = cv2.resize(frame, self.target_size)

            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(
                    str(self.video_path), fourcc, 10.0, self.target_size
                )

            if self.video_writer:
                self.video_writer.write(frame)

            timestamp = time.time()
            if self.csv_writer:
                self.csv_writer.writerow(
                    {
                        "timestamp": f"{timestamp:.3f}",
                        "frame_id": self.frame_count,
                        "speed_motor": f"{self.last_speed:.2f}",
                        "steer_motor": f"{self.last_steer:.2f}",
                    }
                )

            self.frame_count += 1
        except Exception as exc:
            if self.debugging:
                self.logger.error(f"DataCollector frame error: {exc}")
            self._send_status("error", {"message": str(exc)})

    @staticmethod
    def _decode_frame(frame_b64: str):
        try:
            frame_bytes = base64.b64decode(frame_b64)
            frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    # Filesystem helpers ----------------------------------------------------
    def _close_outputs(self):
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None

        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

    def _create_zip(self):
        if not self.session_dir:
            return

        zip_path = self.session_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.session_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.session_dir)
                    zipf.write(file_path, arcname)
        self.zip_path = zip_path

    # Status ----------------------------------------------------------------
    def _send_status(self, state: str, extra: Optional[dict] = None):
        payload = {"state": state}
        if extra:
            payload.update(extra)
        self.statusSender.send(payload)

    # Shutdown --------------------------------------------------------------
    def stop(self):
        self._close_outputs()
        super(processDataCollector, self).stop()
