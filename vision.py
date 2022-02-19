# Imports:
from networktables import NetworkTables
#from asyncio.windows_events import NULL
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard")

cap = cv2.VideoCapture(2)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

reduceF = 1.0

while True:
    channel = sd.getNumber("color channel", 0)
    frame = []
    ret, frame = cap.read()
    #frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    small = cv2.resize(frame, None, fx=1/reduceF, fy=1/reduceF)
    blur = cv2.medianBlur(frame, 9)
    red = blur[:,:,channel]
    blue = blur[:,:,channel]
    #ret, red = cv2.threshold(red, 200, 255, cv2.THRESH_BINARY_INV)
    output = small.copy()
    circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, 200, param1=80, param2=40)
    found = 0
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            if(x >= frame.shape[0]-1 or y >= frame.shape[1]-1):
                continue

            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            if x < r or y < r or x > frame.shape[1] - r or y > frame.shape[0] - r:
                continue

            x = int(x / reduceF)
            y = int(y / reduceF)
            r = int(r / reduceF)

            circ = np.zeros((2*r, 2*r, 3), dtype="uint8")
            cv2.circle(circ, (r, r), r, (255, 255, 255), thickness=cv2.FILLED)
            rectframe = small[y-r:y+r, x-r:x+r]
            cframe = cv2.bitwise_and(circ, rectframe)
            #cv2.imshow("output", cframe)
            #c = cv2.waitKey(0)
            rsum = int(np.sum(cframe[:,:,2]))
            bsum = int(np.sum(cframe[:,:,0]))
            gsum = int(np.sum(cframe[:,:,1]))
            tsum = rsum + bsum + gsum
            sums = [bsum, gsum, rsum]
            if tsum == 0:
                continue
            print(str(sums[channel]) + ' ' + str(tsum))
            if sums[channel] / tsum > [0.43, 1.00, 0.59][channel]:
                cv2.putText(output, str(round(sums[channel]/tsum, 3)), (x-r-5, y-r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
                cv2.circle(output, (x, y), r, (0, 0, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                cv2.rectangle(output, (x-r, y-r), (x+r, y+r), (0, 0, 255), 5)
                found = 1

        # show the output image
    cv2.imshow("output", output)
    sd.putNumber("found on channel " + str(channel), found)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()