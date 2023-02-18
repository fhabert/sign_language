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
    model = Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3,3), strides=1, activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="valid"))
    model.add(layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="valid"))
    model.add(layers.Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(26, activation="softmax"))
    return model

def get_output(n, features, test):
    output = 0
    reshape_feature = np.reshape(features, (1, 28, 28, 1))
    outputs = n.predict(reshape_feature)
    final_output = round(np.argmax(np.array(outputs[0])))
    print(outputs)
    print(final_output)
    print(test)
    if final_output == test:
        output = 1
    return final_output, output

data_test = pd.read_csv("dataset\sign_mnist_test.csv", sep=";", encoding="utf-8")
df_test = pd.DataFrame(data_test)
data_train = pd.read_csv("dataset\sign_mnist_train.csv", sep=";", encoding="utf-8")
df_train = pd.DataFrame(data_train)

df_train.drop_duplicates(subset='label').sort_values(by='label')

Y_train = df_train.iloc[:, 0].to_numpy()
X_train = df_train.iloc[:, 1:].to_numpy().reshape(-1, 28, 28)
Y_test = df_test.iloc[:, 0].to_numpy()
X_test = df_test.iloc[:, 1:].to_numpy().reshape(-1, 28, 28)

# print("Getting the model")
# model = cnn_model()
# optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
# model.fit(X_train, Y_train, batch_size=16, shuffle=True, epochs=4, validation_data=(X_test, Y_test))

print("All good")

# model.save_weights('./my_checkpoint')
# results = []
# for i in range(10):
#     features = X_test[i]
#     lab = Y_test[i]
#     result, output = get_output(model, features, lab)
#     results.append(output)

# print("Percentage of success:", sum(results)/len(results))
