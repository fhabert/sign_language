import cv2 as cv
import numpy as np
import nn_sign
import matplotlib.pyplot as plt
from PIL import Image
import time
import math

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv.CAP_PROP_BRIGHTNESS, 0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

hand_pixels = []
word = []
colors = {"blue": (255, 0, 0), "green": (0,255,0)}

model = nn_sign.create_model()
model.load_weights("./my_checkpoint")

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
last_frame = None
threshold_pix = 5

lines_coor = [[(x_start, y_start), (x_end, y_start)], [(x_start, y_end), (x_end, y_end)], 
            [(x_start, y_start), (x_start, y_end)], [(x_end, y_start), (x_end, y_end)]]

hand_present = False
start_time = time.time()
movement_start_time = time.time()

last_frame_time = 0
fps = 1

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
    if time.time() - start_time < 4:
        continue
    frame_full = np.array(frame_full)
    frame = frame_full[y_start : y_end, x_start : x_end, : ]
    frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
    if time.time() - last_frame_time > 1 / fps:
        last_frame_time = time.time()

        if background_pix is None:
            background_pix = frame
            pixels_value = np.array(background_pix).flatten()
    
        print("back: ", np.mean(background_pix))
        print("frame: ", np.mean(frame))
        # print("sub: ", np.mean(frame - background_pix, axis=None))
        if last_frame is not None:
            diff = np.abs(np.mean(frame - background_pix, axis=None) - np.mean(last_frame - background_pix, axis=None))
            print("diff: ", diff)
            if diff > threshold_pix and (time.time() - movement_start_time > 1 or not hand_present):
                hand_present = not hand_present
                movement_start_time = time.time()
        last_frame = frame
        # if np.mean(background_pix, axis=)

        

        if cv.waitKey(1) == ord('t'):
            hand_pixels = get_pixels_hands(frame)
            img = Image.fromarray(np.uint8(hand_pixels))
            img_resize = img.resize((28,28))
            img_grey_scale = [[np.uint8(sum(x)/3) for x in inner_list] for inner_list in np.asarray(img_resize)]
            # img_grey = Image.fromarray(np.uint8(img_grey_scale), "L")
            img_resize.show()
            pixels = np.array(img_grey_scale).flatten()
            normalize_func = np.vectorize(lambda t: t * 1/255)
            pixels_norm = normalize_func(np.array(pixels))
            input = np.asfarray(pixels_norm)
            formated_input = [[x for x in input]]
            outputs = model.predict(formated_input)
            final_output = round(np.argmax(np.array(outputs[0])))
            if len(word) == 0:
                word.append(decoding[final_output])
            else:
                word.append(decoding[final_output].lower())
            time.sleep(2)
            print("The sign is:", decoding[final_output])


print("The word was:", ("").join(word))
cap.release()
cv.destroyAllWindows()




    
# li = np.asarray(list(nn_sign.df_test.iloc[0][1:])).reshape((28,28))
# img_array = Image.fromarray(np.asarray(li), "L")
# img_grey = img_array.show()