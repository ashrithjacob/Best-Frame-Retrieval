#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:24:03 2022

@author: ashrith
"""
# Importing all necessary libraries
import cv2
import os

# Variables in CAPS are user defined
PATH_TO_VIDEO = "./videos/"
FILE = "V5"
vid = cv2.VideoCapture(PATH_TO_VIDEO + str(FILE) + ".mp4")
store = "./videos/video_data/"+FILE
frame_rate = 30 #number of frames to skip when saving
try:
    # creating a folder named data
    if not os.path.exists(store):
        os.makedirs(store)

# if not created then raise error
except OSError:
    print("Error: Creating directory for frame storage:", str(store))

# frame
currentframe = 0

while True:
    # reading from frame
    success, frame = vid.read()

    if success:
        # Extracting every 'frame_rate' number of frames
        if currentframe % frame_rate == 0:
            # continue creating images until video remains
            name = (
                str(store)
                + "/frame"
                + "_"
                + str(FILE)
                + "_"
                + str(currentframe)
                + ".jpg"
            )
            print("Creating..." + name)
            # writing the extracted images
            cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
vid.release()
cv2.destroyAllWindows()
