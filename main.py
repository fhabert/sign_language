import cv2 as cv
import numpy as np
import nn_sign
import matplotlib.pyplot as plt
from PIL import Image

cap = cv.VideoCapture(0)
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    size = frame.shape
    lines_coor = [[(x_start, y_start), (x_end, y_start)], [(x_start, y_end), (x_end, y_end)], 
                  [(x_start, y_start), (x_start, y_end)], [(x_end, y_start), (x_end, y_end)]]
    for item in lines_coor:
        cv.line(img=frame, pt1=item[0], pt2=item[1], color=colors["blue"], thickness=1, lineType=8, shift=0)
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    elif cv.waitKey(1) == ord('m'):
        print("#########")
        pixels_value = np.array(hand_pixels).flatten()
        plt.hist(pixels_value, bins=20, alpha=0.7)
        plt.show()
    elif cv.waitKey(1) == ord('t'):
        hand_pixels = []
        for i in range(y_start, y_end+1):
            hand_pixels.append(frame[i][x_start:x_end+1])
        img = Image.fromarray(np.uint8(hand_pixels))
        img_resize = img.resize((28,28))
        img_grey_scale = [[np.uint8(sum(x)/3) for x in inner_list] for inner_list in np.asarray(img_resize)]
        pixels = np.array(img_grey_scale).flatten()
        normalize_func = np.vectorize(lambda t: t ** 1/255)
        pixels_norm = normalize_func(np.array(pixels))
        input = np.asfarray(pixels_norm)
        formated_input = [[x for x in input]]
        outputs = model.predict(formated_input)
        final_output = round(np.argmax(np.array(outputs[0])))
        print(word.append(decoding[final_output]))
        print("The sign is:", decoding[final_output])

print("The word was:", ("").join(word))
cap.release()
cv.destroyAllWindows()