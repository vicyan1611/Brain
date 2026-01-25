from gpiozero import DistanceSensor
from time import sleep
import statistics

sensor = DistanceSensor(echo=23, trigger=24, max_distance=3.0)

def get_distance(n=5):
    samples = [sensor.distance for _ in range(n)]
    distance_m = statistics.median(samples)
    return round(distance_m * 100, 2)

