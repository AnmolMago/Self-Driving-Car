import argparse
import base64
import json
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from datetime import datetime
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import pdb
import cv2
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

save_img = False
def preprocess(img):
    global save_img
    if save_img:
        cv2.imwrite('./thisiswhatiget.jpg', img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)[60:140]
    img = cv2.resize(img,(160, 48), interpolation=cv2.INTER_AREA)
    if save_img:
        cv2.imwrite('./thisiswhatioutput.jpg', img)
    save_img = False
    return img/255.

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = np.asarray(image)
    original_image = image.copy()
    image = preprocess(image)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    try:
        steering_angle = float(model.predict(image[None, :, :, :], batch_size=1))
    except Exception as e:
        print(e)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    
    throttle = max(-1,1 - (speed / 15.0))
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

    # save frame
    if args.image_folder != '':
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(args.image_folder, timestamp)
        # image.save('{}.jpg'.format(image_filename))
        cv2.imwrite('{}.jpg'.format(image_filename), cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB))

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)