import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from PIL import Image

# def get_my_model():
#     my_model = Sequential()
#     input = InputLayer(input_shape=(784, ))
#     my_model.add(input)
#     my_model.add(Dense(200, activation="tanh"))
#     my_model.add(Dense(24, activation="tanh"))
#     opt = Adam(learning_rate=0.0005)
#     my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)
#     my_model.fit(np.asarray(features_train).astype('float64'), np.asarray(labels_train).astype('float64'), epochs=50, batch_size=16)
#     return my_model

def cnn_model():
    my_model = Sequential()
    my_model.add(layers.Conv2D(32, kernel_size=3, strides=1, activation="relu", input_shape=(28, 28, 1), padding="valid"))
    my_model.add(layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    my_model.add(layers.Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="valid"))
    my_model.add(layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    my_model.add(layers.Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="valid"))
    my_model.add(layers.Flatten())
    my_model.add(Dense(64, activation="relu"))
    my_model.add(Dense(24, activation="sigmoid"))
    my_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    # my_model.fit(features_train.astype('float64'), labels_train.astype('float64'), epochs=10, \
    #                 validation_data=(features_test.astype("float64"), labels_test.astype("float64")))
    return my_model

def set_dataframe(df):
    labels_df = sorted(list(df.label.unique()))
    normalize_func = np.vectorize(lambda t: t * 1/255)
    x_data_norm = []
    for i in range(len(df)):
        row = normalize_func(np.array(df.iloc[i][1:]))
        x_data_norm.append(np.array(row).reshape((28,28)))
    one_hot_titles = []
    for i in range(len(labels_df)):
        category = [0.2 if i != j else 0.8 for j in range(len(labels_df))]
        one_hot_titles.append(category)
    labels_organized = []
    for i in range(len(df)):
        nb = df.iloc[i][0]
        labels_organized.append(one_hot_titles[nb-1])
    return np.asarray(x_data_norm), np.asarray(labels_organized)

def get_output(n, features, test):
    output = 0
    reshape_feature = np.reshape(features, (1, 28, 28, 1))
    outputs = n.predict(reshape_feature)
    final_output = round(np.argmax(np.array(outputs[0])))
    output_test = round(np.argmax(np.array(test)))
    print(outputs)
    print(final_output)
    print(output_test)
    if final_output == output_test:
        output = 1
    return final_output, output   

# data_test = pd.read_csv("dataset\sign_mnist_test.csv", sep=";", encoding="utf-8")
# df_test = pd.DataFrame(data_test)
# data_train = pd.read_csv("dataset\sign_mnist_train.csv", sep=";", encoding="utf-8")
# df_train = pd.DataFrame(data_train)
# features_test, labels_test = set_dataframe(df_test)
# features_train, labels_train = set_dataframe(df_train) 

print("Getting the model")
# model = cnn_model()
print("All good")

# model.save_weights('./my_checkpoint')
# results = []
# for i in range(10):
#     features = features_test[i]
#     lab = labels_test[i]
#     result, output = get_output(model, features, lab)
#     results.append(output)

# print("Percentage of success:", sum(results)/len(results))
