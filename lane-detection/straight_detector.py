import numpy as np
import pandas as pd
import cv2

from moviepy import editor
import moviepy

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    """
    # create an array of the same size as of the input image 
    mask = np.zeros_like(image)  

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.125, rows * 1]
    top_left     = [cols * 0.45, rows * 0.4]
    bottom_right = [cols * 1, rows * 1]
    top_right    = [cols * 0.7, rows * 0.4]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: grayscale image which should be an output from the edge detector
    """
    # Distance resolution of the accumulator in pixels.
    rho = 1              
    # Angle resolution of the accumulator in radians.
    theta = np.pi/180    
    # Only lines that are greater than threshold will be returned.
    threshold = 150       
    # Line segments shorter than that are rejected.
    minLineLength = 150   
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500     
    # function returns an array containing dimensions of straight lines 
    # appearing in the input image
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)
    
def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    # 
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
  
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
  
def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image (video frame in our case).
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness. 
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
    """
    Process the input frame to detect lane lines.
    Parameters:
        image: image of a road where one wants to detect lane lines
        (we will be passing frames of video to this function)
    """
    # convert the RGB image to Gray scale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5 
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    low_t = 150
    high_t = 200

    edges = cv2.Canny(blur, low_t, high_t)
    
    region = region_selection(edges)

    hough = hough_transform(region)

    # lastly we draw the lines on our resulting frame and return it as output 
    result = draw_lane_lines(image, lane_lines(image, hough))
    
    return result

# driver function
def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
    """
    input_video = editor.VideoFileClip(test_video, audio=False)

    # apply the function "frame_processor" to each frame of the video
    # will give more detail about "frame_processor" in further steps
    # "processed" stores the output video
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_video, audio=False)
    

# calling driver function
input_path = 'input.mp4'
output_path = 'output.mp4'

process_video(input_path, output_path)
