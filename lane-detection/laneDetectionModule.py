import cv2
import numpy as np
import utils

curveList = []
avgVal = 10

def getLaneCurve(img, display=2):
    imgCopy = img.copy()
    imgResult = img.copy() # Initialize here to ensure it exists for all paths
    
    # Step 1: Thresholding
    imgThres = utils.thresholding(img)

    # Step 2: Warp Image
    h, w, c = img.shape
    points = utils.valTrackbars()
    
    # WARPING LOGIC: 
    # 1. Warp the Threshold image for Calculation (Clean black/white data)
    imgWarp = utils.warpImg(imgThres, points, w, h)
    # 2. Warp the Color image for Display (Visual reference)
    imgWarpPoints = utils.drawPoints(imgCopy, points)
    
    # Step 3: Histogram Calculation
    # Note: using imgWarp (which is now the thresholded version)
    middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    # Step 4: Averaging
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(np.mean(curveList))

    # Step 5: Display
    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, points, w, h, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:h // 3, 0:w] = 0, 0, 0
        
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        
        midY = 450
        cv2.putText(imgResult, str(curve), (w // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        
        cv2.line(imgResult, (w // 2, midY), (w // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, (w // 2 + (curve * 3), midY - 25), (w // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)

        wGap = w // 20 
        for x in range(-30, 30):
            cv2.line(imgResult, (wGap * x + int(curve // 50), midY - 10),
                     (wGap * x + int(curve // 50), midY + 10), (0, 0, 255), 2)

    if display == 2:
        imgWarpBGR = cv2.cvtColor(imgWarp, cv2.COLOR_GRAY2BGR)
        
        imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarpBGR],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)
    
    return curve

if __name__ == '__main__':
    cap = cv2.VideoCapture('input.mp4')
    initialTracebarVals = [102, 80, 20, 214] 
    utils.initializeTrackbars(initialTracebarVals)
    
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        if not success: break 
        
        img = cv2.resize(img, (480, 240))
        curve = getLaneCurve(img)
        
        cv2.waitKey(1)
        