import cv2 as cv
import numpy as np
import nn_sign
from pynput import keyboard
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def on_press(key):
    if key == keyboard.Key.esc:
        return False
    elif key == keyboard.Key.up:
        print("#########")
        print(hand_pixels)
        pixels_value = np.array(hand_pixels).flatten()
        plt.hist(pixels_value, bins=20, alpha=0.7)
        plt.show()
    pass

hand_pixels = []
word = []
listener = keyboard.Listener(on_press=on_press)
listener.start()
colors = {"blue": (255, 0, 0), "green": (0,255,0)}
# model = nn_sign.model()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    size = frame.shape
    lines_coor = [[(50, 150), (350, 150)], [(50, 450), (350, 450)], 
                  [(50, 150), (50, 450)], [(350, 150), (350, 450)]]
    for item in lines_coor:
        cv.line(img=frame, pt1=item[0], pt2=item[1], color=colors["blue"], thickness=4, lineType=8, shift=0)
    hand_pixels = frame[50:350, 150:450]
    # value = model.get_output(model, hand)
    # word.append(value)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# print(word)
cap.release()
cv.destroyAllWindows()