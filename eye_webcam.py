#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:59:41 2019

@author: eadali
"""

import cv2
import dlib
from imutils import face_utils
from numpy import amax, amin, zeros, uint8, ones, int64
from scipy.spatial.distance import euclidean




# CONFIGS
# =============================================================================
eye_predictor_path = 'eye_predictor.dat'
ear_threshold = 0.2
# =============================================================================

# CAMERA LOOP
# =============================================================================
# loads eye shape predictor
eye_predictor = dlib.shape_predictor(eye_predictor_path)

# creates opencv camera object
webcam = cv2.VideoCapture(0)

while(webcam.isOpened()):
    # captures frame by frame
    _, frame = webcam.read()

    # converts colored frame to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gets eye landmarks
    frame_rectangle = dlib.rectangle(left=0, top=0,
                                     right=frame.shape[1], bottom=frame.shape[0])
    landmarks = face_utils.shape_to_np(eye_predictor(gray_frame, frame_rectangle))

    # calculates ear
    num = euclidean(landmarks[1], landmarks[5]) + euclidean(landmarks[2], landmarks[4])
    den = euclidean(landmarks[0], landmarks[3])
    ear = num / (2.0 * den)


    if ear > ear_threshold:

        # crop roi
        min_x = amin(landmarks[:,0])
        min_y = amin(landmarks[:,1])
        max_x = amax(landmarks[:,0])
        max_y = amax(landmarks[:,1])

        roi_frame = cv2.GaussianBlur(gray_frame[min_y:max_y, min_x:max_x], (5,5), 0)

        # calculates binary image for pupil detection
        _, binary_frame = cv2.threshold(roi_frame, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # masks frame for eye
        roi_landmarks = zeros(landmarks.shape, int64)
        roi_landmarks[:,0] = landmarks[:,0] - min_x
        roi_landmarks[:,1] = landmarks[:,1] - min_y

        mask = zeros(roi_frame.shape, uint8)
        cv2.fillPoly(mask, pts=[roi_landmarks], color=1)
        binary_frame = binary_frame * mask

        # erodes image for remove eyelids
        kernel = ones((5,5), uint8)
        binary_frame = cv2.erode(binary_frame,kernel,iterations = 1)

        # finds contours
        _, contours, _ = cv2.findContours(binary_frame, 1, 2)

        iris_center = None
        iris_radius = None
        max_area = 0

        # selects largest contour
        for single_contour in contours:
            area = cv2.contourArea(single_contour)

            if (single_contour.shape[0] > 2.5) and (area > max_area):
                    (x,y), radius = cv2.minEnclosingCircle(single_contour)
                    iris_center = (int(x), int(y))
                    iris_radius = int(radius)
                    max_area = area

        # draws contour on frame
        if iris_center is not None:
            if iris_radius is not None:
                iris_center = (iris_center[0]+min_x, iris_center[1]+min_y)
                cv2.circle(frame, iris_center, iris_radius,(0,255,0), 2)

    # draws eye landmarks
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# when everything done, release the capture
webcam.release()
cv2.destroyAllWindows()
# =============================================================================
