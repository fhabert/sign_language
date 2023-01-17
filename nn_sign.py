import pandas as pd
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.layers import Dense
import numpy as np
import csv
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
from PIL import Image

def get_my_model():
    my_model = tf.keras.models.Sequential()
    input = tf.keras.layers.InputLayer(input_shape=(1784, ))
    my_model.add(input)
    my_model.add(tf.keras.layers.Dense(24, activation='relu'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    my_model.fit(np.asarray(features_train).astype('float64'), np.asarray(labels_train).astype('float64'), epochs=100, batch_size=5)
    return my_model

def set_dataframe(df):
    labels_df = sorted(list(df.label.unique()))
    normalize_func = np.vectorize(lambda t: t ** 1/255)
    x_data_norm = []
    for i in range(len(df)):
        row = normalize_func(np.array(df.iloc[i][1:]))
        x_data_norm.append(row)
    one_hot_titles = []
    for i in range(len(labels_df)):
        category = [0.2 if i != j else 0.8 for j in range(len(labels_df))]
        one_hot_titles.append(category)
    labels_organized = []
    for i in range(len(df)):
        nb = df.iloc[i][0]
        labels_organized.append(one_hot_titles[nb-1])
    return x_data_norm, labels_organized

data_test = pd.read_csv("dataset\sign_mnist_test.csv", sep=";", encoding="utf-8")
df_test = pd.DataFrame(data_test)
data_train = pd.read_csv("dataset\sign_mnist_train.csv", sep=";", encoding="utf-8")
df_train = pd.DataFrame(data_test)
features_test, labels_test = set_dataframe(df_test)
features_train, labels_train = set_dataframe(df_train) 

model = get_my_model()
# result = get_value(model, 0)
