import os
import cv2
import csv
import time
import threading
import zipfile
from multiprocessing import Process, Queue
from datetime import datetime

class DataRecorder(Process):
    def __init__(self, video_queue, data_queue, output_dir="temp/data_collection"): 
        super().__init__()
        self.video_queue = video_queue
        self.data_queue = data_queue
        self.output_dir = output_dir
        self.recording = False
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None
        self.file_prefix = None
        self.frames = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_prefix = f"record_{timestamp}"
        video_path = os.path.join(self.output_dir, f"{self.file_prefix}.avi")
        csv_path = os.path.join(self.output_dir, f"{self.file_prefix}.csv")
        self.video_writer = None
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "speed", "steering"])
        self.recording = True
        self.frames = 0
        return video_path, csv_path

    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
        if self.csv_file:
            self.csv_file.close()
        # Zip the files
        video_path = os.path.join(self.output_dir, f"{self.file_prefix}.avi")
        csv_path = os.path.join(self.output_dir, f"{self.file_prefix}.csv")
        zip_path = os.path.join(self.output_dir, f"{self.file_prefix}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(video_path, os.path.basename(video_path))
            zipf.write(csv_path, os.path.basename(csv_path))
        return zip_path

    def run(self):
        video_path, csv_path = None, None
        while True:
            if not self.recording:
                cmd = self.data_queue.get()
                if cmd == "start":
                    video_path, csv_path = self.start_recording()
                elif cmd == "exit":
                    break
                continue
            try:
                frame, speed, steering = self.video_queue.get(timeout=1)
                now = time.time()
                if self.video_writer is None:
                    h, w, _ = frame.shape
                    self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (w, h))
                self.video_writer.write(frame)
                self.csv_writer.writerow([now, speed, steering])
                self.frames += 1
            except Exception:
                pass
            # Check for stop command
            if not self.data_queue.empty():
                cmd = self.data_queue.get()
                if cmd == "stop":
                    self.stop_recording()
                elif cmd == "exit":
                    break

# Example usage:
# video_queue = Queue()
# data_queue = Queue()
# recorder = DataRecorder(video_queue, data_queue)
# recorder.start()
# data_queue.put("start")
# video_queue.put((frame, speed, steering))
# data_queue.put("stop")
# data_queue.put("exit")
