import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense

def generate_data():
    # Main Driving Log
    with open('./Training/driving_log.csv') as file:
    data = csv.reader()
    for entry in data:
        yield({}, {'output': })
    



model = Sequential()
model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)

