import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import math

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv.CAP_PROP_AUTO_WB, 1)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25) #manual mode
cap.set(cv.CAP_PROP_EXPOSURE, -4)


if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    if cv.waitKey(1) == ord('q'):
        break
    ret, frame_full = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

cap.release()
cv.destroyAllWindows()