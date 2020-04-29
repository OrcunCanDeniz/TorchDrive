import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
import utils # helpers

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)

model = None
prev_image_array = None

if torch.cuda.is_available():
    print('Using GPU !!!')
    device = torch.device("cuda:0")# choose GPU number 0, as computation device
    torch.backends.cudnn.benchmark = True #CUDNN autotuner to find best algorithm for present hardware
else:
    print('Using CPU !!!')
    device = torch.device("cpu")# choose CPU as computation device

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current speed of the car
        speed = torch.tensor(float(data["speed"])).unsqueeze(0).to(device)
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array
            image = torch.from_numpy(image).to(device).permute(0,3,1,2)
            # predict the steering angle for the image
            #steering_angle, throttle, reverse = model((image,speed.unsqueeze(0)))
            out = model((image,speed.unsqueeze(0)))
            steering_angle = float(out[0])

            throttle = float(out[1]) if out[1] > abs(out[2]) else -float(out[2]) #Get the most likely one between forward and reverse throttle

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


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

    #load model
    model = torch.load(args.model)
    model.eval()

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
