import cv2
import time

cam = cv2.VideoCapture(-1)
ret, frame = cam.read()

while True:
    ret, frame = cam.read()
    print(frame.min(), frame.max())
    # show
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
