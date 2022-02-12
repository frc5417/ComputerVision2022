# Imports:
#import networktables
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    red = cv2.medianBlur(frame.copy(), 7)[:,:,2]
    #ret, red = cv2.threshold(red, 200, 255, cv2.THRESH_BINARY_INV)
    output = red.copy()
    circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=60)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
    cv2.imshow("output", output)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()