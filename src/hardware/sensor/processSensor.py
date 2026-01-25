from src.templates.workerprocess import WorkerProcess
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import DistanceReading
import statistics
import time

try:
	from gpiozero import DistanceSensor
except ImportError:
	DistanceSensor = None  # allows the module to import on non-RPi machines

class DistanceReader(ThreadWithStop):
	"""Polls a gpiozero DistanceSensor and publishes median-filtered distance."""

	def __init__(self, queuesList, logger=None, pause=0.05, samples=5, max_distance=3.0):
		super().__init__(pause=pause)
		self.logger = logger
		self.samples = samples
		self.sender = messageHandlerSender(queuesList, DistanceReading)

		# Initialize the hardware sensor only if available
		if DistanceSensor is None:
			raise RuntimeError("gpiozero not available; run on Raspberry Pi or install gpiozero")
		# BCM pins: echo=23, trigger=24 by default (as provided by the user)
		self.sensor = DistanceSensor(echo=23, trigger=24, max_distance=max_distance)

	def thread_work(self):
		try:
			# Collect a few samples and median-filter
			distances = [self.sensor.distance for _ in range(self.samples)]
			distance_m = statistics.median(distances)
			distance_cm = round(distance_m * 100, 2)
			# Send to queue
			self.sender.send(distance_cm)

			if self.logger:
				self.logger.debug("DistanceReader: %.2f cm", distance_cm)
		except Exception as e:
			if self.logger:
				self.logger.error("DistanceReader error: %s", e)

class processSensor(WorkerProcess):
	"""Process that starts a distance reader thread and publishes readings."""

	def __init__(self, queueList, logging, ready_event=None, debugging=False):
		self.queuesList = queueList
		self.logging = logging
		self.debugging = debugging
		super(processSensor, self).__init__(self.queuesList, ready_event)

	def _init_threads(self):
		# Run the sensor thread at ~20 Hz (pause=0.05). Adjust as needed.
		try:
			sensor_thread = DistanceReader(self.queuesList, self.logging, pause=0.05)
		except Exception as e:
			if self.logging:
				self.logging.error("Failed to start DistanceReader: %s", e)
			return
		self.threads.append(sensor_thread)
