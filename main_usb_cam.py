# Emotion detector for Raspberry Pi 4
# Author: Elliot Blanford
# Date: 1/18/2021
# Description: Just run it and make faces at the camera! It will print out predictions and
# confidence if it is above threshold. It will also buzz on detecting a change in emotional state

# Original inspiration by Evan Juras
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
# I updated it to work with tensorflow v2, changed it to an emotion detection model, and added feedback device
# a vibrating motor controlled by GPIO pins

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow.compat.v1 as tf
import argparse
import sys
from PIL import Image
import RPi.GPIO as GPIO
import time
import tflite_runtime.interpreter as tflite
#import numba

cascPath = "/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Set up camera constants
IM_WIDTH = 640//2 
IM_HEIGHT = 480//2 #720//4


# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# translate model output to label
mapper = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

camera = cv2.VideoCapture(0)

#set up GPIO
GPIO.setmode(GPIO.BOARD)
buzzer_pin = 8
GPIO.setup(buzzer_pin, GPIO.OUT)

#these two variables will be used to track repeated inference of the same emotion
prev_emo_state = 6 #start with neutral
repeats = 0
confidence_threshold = 70 #in %
repeat_threshold = 3 #number of times an emotion is detected before emotion state is updated

while camera.isOpened():
    ret, frame1 = camera.read()
    t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = cv2.resize(frame1, (IM_WIDTH, IM_HEIGHT))
    frame.setflags(write=1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = frame_gray
    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        box_size = max(h, w)
        roi_gray = frame_gray[y:y+box_size, x:x+box_size]
        roi_color = frame[y:y+box_size, x:x+box_size]

    face_gray = cv2.resize(roi_gray, (48,48))
    face_expanded = np.expand_dims(face_gray/255, axis=2).astype('float32')

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="emotion_quarter_size.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], [face_expanded])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    confidence = np.max(output_data[0]) * 100
    # need to show predition on the screen, if it's a 'confident' prediction, i'll show the %
    if confidence > (confidence_threshold - 20):
        emo_state = np.where(output_data[0] == np.max(output_data[0]))[0][0]
        print("Guess: ", mapper[emo_state],
              "(%.02f%%)" % confidence)
        if emo_state == prev_emo_state and confidence > confidence_threshold:
            #count how long you've been on this state
            repeats += 1
            #print('repeat', repeats)
        if repeats >= repeat_threshold and emo_state != prev_emo_state and confidence > confidence_threshold:
            #alert user to change and reset count
            print('State change detected!')
            #GPIO.output(buzzer_pin, True)
            #time.sleep(0.1)
            #GPIO.output(buzzer_pin, False)
            #time.sleep(0.2)
            #GPIO.output(buzzer_pin, True)
            #time.sleep(0.1)
            #GPIO.output(buzzer_pin, False)
            repeats = 0
        if confidence > confidence_threshold:
            prev_emo_state = emo_state
    else:
        print("Guess: ", mapper[np.where(output_data[0] == np.max(output_data[0]))[0][0]])

    cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (10, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Emotion detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
GPIO.cleanup()

cv2.destroyAllWindows()

