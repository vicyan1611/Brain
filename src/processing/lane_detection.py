import cv2
import numpy as np
from collections import deque

# Base dimensions used by the original tuning values
_BASE_W = 480
_BASE_H = 240

# Default source points (from original trackbar defaults for 480x240)
DEFAULT_SRC_POINTS = [
    (102, 80),   # top-left
    (_BASE_W - 102, 80),  # top-right
    (20, 214),   # bottom-left
    (_BASE_W - 20, 214),  # bottom-right
]


def _threshold_white(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 55, 255])
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    return mask_white


def _warp_img(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))


def _get_histogram(img, min_per=0.1, region=1):
    if region == 1:
        hist_values = np.sum(img, axis=0)
    else:
        hist_values = np.sum(img[img.shape[0] // region :, :], axis=0)

    max_value = np.max(hist_values)
    if max_value == 0:
        return 0, None

    min_value = min_per * max_value
    index_array = np.where(hist_values >= min_value)
    base_point = int(np.average(index_array))
    return base_point, hist_values


def _scale_points(points, w, h):
    """Scale the default source points to the current frame size."""
    xs = [p[0] * (w / _BASE_W) for p in points]
    ys = [p[1] * (h / _BASE_H) for p in points]
    return list(zip(xs, ys))


class LaneCurveEstimator:
    """Extracts a lane curvature indicator from a frame using histogram warping.

    The returned curve is an integer; negative values mean turn left, positive turn right.
    """

    def __init__(self, points=None, avg_window=10):
        self.points = points if points is not None else DEFAULT_SRC_POINTS
        self.avg_window = avg_window
        self.curve_buffer = deque(maxlen=avg_window)

    def process(self, frame):
        h, w = frame.shape[:2]
        src_points = _scale_points(self.points, w, h)

        img_thresh = _threshold_white(frame)
        img_warp = _warp_img(img_thresh, src_points, w, h)

        middle_point, _ = _get_histogram(img_warp, min_per=0.5, region=4)
        curve_avg_point, _ = _get_histogram(img_warp, min_per=0.9)
        curve_raw = curve_avg_point - middle_point

        self.curve_buffer.append(curve_raw)
        curve = int(np.mean(self.curve_buffer))
        return curve
