import csv
import cv2
import json
from random import shuffle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

BATCH_SIZE = 512
EPOCHS = 7

def load_data():
    data = []
    with open('./Training/driving_log.csv') as logs:
        img_col, steering_col = 0, 3
        for log in csv.reader(logs):
            data.append((log[img_col], float(log[steering_col])))

    shuffle(data)
    split = int(len(data) * 0.8)
    validation_data = data[split:]
    training_data = data[:split]

    return training_data, validation_data

def preprocess(img):
    img = img[32:135, 0:320]
    img = cv2.resize(img,(64,64), interpolation=cv2.INTER_AREA)    
    return img

def flipped(img):
    return cv2.flip(img,1)

def generate_data(data):
    while True:
        for i in range(0, len(data), BATCH_SIZE//2):
            end = min(i+BATCH_SIZE//2, len(data))
            batch_len, batch_index = (end-i)*2, 0

            batch_image = np.zeros((batch_len, 64, 64, 3))
            batch_label = np.zeros(batch_len)

            for j in range(i, end):
                img, steering = data[j]
                img = preprocess(cv2.imread(img))
                batch_image[batch_index] = img
                batch_label[batch_index] = steering
                batch_image[batch_index + 1] = flipped(img)
                batch_label[batch_index + 1] = -1 * steering

                batch_index += 2

            yield(batch_image, batch_label)

def get_model():
    # Define Keras model architecture with Sequential
    model = Sequential()

    # Input Convolutional Layer - I/O Match to Use Best Color Space 
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(64,64,3)))

    # Add Non-Linearity
    model.add(Activation('elu'))

    # Add Convolutional Layer Set - 5x5 filter kernal with 2x2 strides (nVidia Architecture)
    model.add(Convolution2D(3, 5, 5, subsample=(2, 2), border_mode="valid", name='layer1_conv2d'))

    # Add Non-Linearity
    model.add(Activation('elu'))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", name='layer2_conv2d'))

    # Add Non-Linearity
    model.add(Activation('elu'))

    model.add(Convolution2D(36, 5, 5, subsample=(1, 1), border_mode="valid", name='layer3_conv2d'))

    # Add Dropout to Reduce Over-fitting
    model.add(Dropout(.5))

    # Add Non-Linearity
    model.add(Activation('elu'))

    # Add Convolutional Layer Set - 3x3 filter kernal with 1x1 strides (nVidia Architecture)
    model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode="valid", name='layer4_conv2d'))

    # Add Non-Linearity
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", name='layer5_conv2d'))

    # Add Dropout to Reduce Over-fitting
    model.add(Dropout(.5))

    # Add Non-Linearity
    model.add(Activation('elu'))

    # Convert to Fully Connected Layer
    model.add(Flatten())

    # Add Dropout to Reduce Over-fitting
    model.add(Activation('elu'))

    # Add Dense Layer Set - 100 > 50 > 10 (nVidia Architecture)
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('elu'))

    model.add(Dense(50))
    model.add(Activation('elu'))

    model.add(Dense(10))

    # Reduce to Single Output / Prediction Value - Use tanh Activation to bound results between -1 and 1.
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model

def train(training_data, validation_data):
    model = get_model()
    model.fit_generator(
        generate_data(training_data),
        samples_per_epoch=len(training_data)*2,
        validation_data=generate_data(validation_data),
        nb_val_samples=len(validation_data)*2,
        nb_epoch=EPOCHS
        )
    return model

def main():
    training_data, validation_data = load_data()
    model = train(training_data, validation_data)
    print("hello")
    with open("./model.json", 'w') as json_file:
        json_file.write(model.to_json())
    print("hello")
    model.save("./model.h5", overwrite=True)


if __name__ == '__main__':
    np.random.seed(200)
    main()