import os
import csv
import json
import random

import cv2
import numpy as np
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

def load_data():
    data = []
    with open('./Training/driving_log.csv') as logs:
        img_col, steering_col = 0, 3
        for log in csv.reader(logs):
            try:
                absoloute = "/Users/Mango/Documents/Development/Learning Plus Challenges/SDC/Term 1/Behavioural Cloning /"
                path = log[img_col].replace(absoloute, './')
                if not os.path.isfile(path):
                    continue
                data.append((path, float(log[steering_col])))
            except ValueError:
                pass

    random.shuffle(data)
    split = int(len(data) * 0.7)
    validation_data = data[split:]
    training_data = data[:split]

    return training_data, validation_data

def preprocess(img, path):
    if img is None:
        print("Null image:", path)
        return None
    img = img[0:100,:,:]
    img = cv2.resize(img,(64,64), interpolation=cv2.INTER_AREA)    
    return img

def flipped(img):
    return cv2.flip(img,1)

def generate_training_data(data):
    while True:
        for i in range(0, len(data), BATCH_SIZE//2):
            end = min(i+BATCH_SIZE//2, len(data))
            batch_len, batch_index = (end-i)*2, 0

            batch_image = np.zeros((batch_len, 64, 64, 3))
            batch_label = np.zeros(batch_len)

            for j in range(i, end):
                img, steering = data[j]
                img = preprocess(np.asarray(cv2.imread(img)), img)
                batch_image[batch_index] = img
                batch_label[batch_index] = steering
                batch_image[batch_index + 1] = flipped(img)
                batch_label[batch_index + 1] = -1 * steering

                batch_index += 2

            yield(batch_image, batch_label)

def generate_validation_data(data):
    while True:
        for i in range(0, len(data), BATCH_SIZE):
            end = min(i+BATCH_SIZE, len(data))

            batch_image = np.zeros((end-i, 64, 64, 3))
            batch_label = np.zeros(end-i)

            for j in range(i, end):
                img, steering = data[j]
                img = preprocess(np.asarray(cv2.imread(img)), img)
                batch_image[j-i] = img
                batch_label[j-i] = steering

            yield(batch_image, batch_label)

def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(64,64,3)))

    model.add(Convolution2D(24,5,5, border_mode='valid', W_regularizer=l2(0.01), activation='elu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5, border_mode='valid', W_regularizer=l2(0.01), activation='elu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5, border_mode='valid', W_regularizer=l2(0.01), activation='elu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.01), activation='elu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.01), activation='elu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer=Adam(lr=LR), loss='mean_squared_error')

    return model

def train(training_data, validation_data):
    checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model = get_model()
    history = model.fit_generator(
        generate_validation_data(training_data),
        samples_per_epoch=len(training_data),
        validation_data=generate_validation_data(validation_data),
        nb_val_samples=len(validation_data),
        nb_epoch=EPOCHS,
        callbacks=[checkpoint],
        verbose=2
        )
    print(history, history.history)
    return model

def save(model):
    print("Starting to save...")
    with open("./model.json", 'w') as json_file:
        json_file.write(model.to_json())
    print("JSON saved")
    model.save("./model_end.h5", overwrite=True)
    print("Save complete")

def main():
    training_data, validation_data = load_data()
    model = train(training_data, validation_data)
    save(model)

def test_images():
    imgs = ["./Training/"+str(i)+".jpg" for i in range(0, 5)]
    images = np.zeros((5, 64, 64, 3))
    for i in range(0,5):
        images[i] = preprocess(np.asarray(cv2.imread(imgs[i])), imgs[i])
    # with open('./model.json', 'r') as json_file:
    #     model_json = json_file.read()
    # model = model_from_json(model_json)
    # model.load_weights("./model_end.h5")
    model = load_model("model_end.h5")
    prediction = model.predict(images)
    print(list(zip(imgs, prediction)))

if __name__ == '__main__':
    np.random.seed(137)
    random.seed(137)
    main()
    test_images()