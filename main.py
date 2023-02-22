import cv2 as cv
import numpy as np
# import nn_sign
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
import tensorflow as tf

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv.CAP_PROP_AUTO_WB, 1)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25) #manual mode
cap.set(cv.CAP_PROP_EXPOSURE, -4)


if not cap.isOpened():
    print("Cannot open camera")
    exit()

hand_pixels = []
word = []
colors = {"blue": (255, 0, 0), "green": (0,255,0)}

model = tf.keras.models.load_model("cnn1")
decoding = [chr(i) for i in range(65, 91)]
x_start, y_start, x_end, y_end = 50, 150, 350, 450

def get_pixels_hands(frame):
    hand_pixels = []
    for i in range(y_start+10, y_end-10):
        hand_pixels.append(frame[i][x_start+10:x_end-10])
    # img_grey_scale = [[np.uint8(sum(x)/3) for x in inner_list] for inner_list in np.asarray(hand_pixels)]
    return hand_pixels

def get_hist_values(li):
    pixels_dic = {}
    for item in li:
        if item in pixels_dic:
            pixels_dic[item] += 1
        else:
            pixels_dic[item] = 1
    return pixels_dic

def show_hist(frame):
    print("######### Here's an histogram to find out the pixel for the skin #######")
    pixels_value = np.array(frame).flatten()
    plt.hist(pixels_value, bins=20, alpha=0.7)
    plt.show()

background_pix = None
start_time = time.time()
threshold_pix = 0.01

lines_coor = [[(x_start, y_start), (x_end, y_start)], [(x_start, y_end), (x_end, y_end)], 
            [(x_start, y_start), (x_start, y_end)], [(x_end, y_start), (x_end, y_end)]]

hand_present = False
last_frame_time = 0
fps = 10
prev_diff = 0

while True:
    if cv.waitKey(1) == ord('q'):
        break
    ret, frame_full = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    for item in lines_coor:
        cv.line(img=frame_full, pt1=item[0], pt2=item[1], color=colors["blue"], thickness=1, lineType=8, shift=0)

        if hand_present:
            for item in lines_coor:
                cv.line(img=frame_full, pt1=item[0], pt2=item[1], color=colors["green"], thickness=1, lineType=8, shift=0)
    
    cv.imshow('frame', frame_full)

    frame_full = np.array(frame_full)
    frame = frame_full[y_start : y_end, x_start : x_end, : ]
    frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]) / 255
    frame_blured = cv.GaussianBlur(frame, (21, 21), cv.BORDER_DEFAULT)

    if time.time() - last_frame_time > 1 / fps:
        last_frame_time = time.time()

        if background_pix is None:
            background_pix = frame_blured

        diff = np.abs(np.mean(frame_blured) - np.mean(background_pix))
        # print("diff: ", diff)
        if diff > threshold_pix:
            hand_present = True
        else:
            hand_present = False
        if diff < threshold_pix and diff > 0.01:
            background_pix = frame_blured
        prev_diff = diff
        if hand_present:
            frame_28 = np.asarray(Image.fromarray(frame).resize((28,28)))
            # reshape_feature = np.reshape(frame_28, (1, 28, 28, 1))
            plt.imshow(frame_28, cmap='gray', vmin=0, vmax=1)
            plt.show()
            frame_28 = frame_28.reshape(1, 28, 28, 1)
            pred = model.predict(frame_28, verbose=0)
            final_output = round(np.argmax(pred[0]))
            print("letter: ", decoding[final_output])
            if len(word) == 0:
                word.append(decoding[final_output])
            else:
                word.append(decoding[final_output].lower())

print("The word was:", ("").join(word))
cap.release()
cv.destroyAllWindows()

