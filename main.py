import cv2 as cv
import numpy as np
import csv
import nn_sign

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

word = []
# model = nn_sign.model()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    size = frame.shape
    lines_coor = [[(50, 150), (350, 150)], [(50, 450), (350, 450)], 
                  [(50, 150), (50, 450)], [(350, 150), (350, 450)]]
    for item in lines_coor:
        cv.line(img=frame, pt1=item[0], pt2=item[1], color=(255, 0, 0), thickness=4, lineType=8, shift=0)
    # Display the resulting frame
    # img = cv.imread(frame)
    hand = frame[50:350, 150:450]
    # value = model.get_output(model, hand)
    # word.append(value)
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
# print(word)
cap.release()
cv.destroyAllWindows()