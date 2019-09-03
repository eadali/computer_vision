#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:59:41 2019

@author: eadali
"""

import dlib
from xml.etree import ElementTree
import cv2
import random
import numpy



def save_eye_landmarks(ibug_landmarks, eye_landmarks):
    tree = ElementTree.parse(ibug_landmarks)
    root = tree.getroot()
    for image in root.iter('image'):
        eye_min_x = 9e9
        eye_min_y = 9e9
        eye_max_x = -1
        eye_max_y = -1

        landmarks = list()

        first_box = True
        for face_box in image.findall('box'):
            if first_box:
                first_box = False


                for landmark in face_box.findall('part'):
                    part_name = int(landmark.get('name'))
                    x = int(landmark.get('x'))
                    y = int(landmark.get('y'))

                    if part_name > 35.5 and part_name < 41.5:
                        landmarks.append([x,y])

                        if x < eye_min_x:
                            eye_min_x = int(max(x - 64 + random.random() * 32, 0))

                        if y < eye_min_y:
                            eye_min_y = int(max(y - 64 + random.random() * 32, 0))

                        if x > eye_max_x:
                            eye_max_x = int(x + 64 - random.random() * 32)

                        if y > eye_max_y:
                            eye_max_y = int(y + 64 - random.random() * 32)

                        landmark.set('name',format(part_name-36,'02d'))

                    else:
                        face_box.remove(landmark)

                face_box.set('top', str(eye_min_y))
                face_box.set('left', str(eye_min_x))
                face_box.set('width', str(eye_max_x-eye_min_x))
                face_box.set('height', str(eye_max_y-eye_min_y))

            else:
                image.remove(face_box)


    tree.write(eye_landmarks)

    return eye_landmarks



# CONFIGS
# =============================================================================
ibug_data_path = 'labels_ibug_300W_train.xml'
eye_data_path = 'eye_train_landmarks.xml'
eye_predictor_path = 'eye_predictor.dat'

# dlib shape predictor parameters
options = dlib.shape_predictor_training_options()
options.tree_depth = 4
options.nu = 0.1
options.cascade_depth = 15
options.feature_pool_size = 400
options.num_test_splits = 50
options.oversampling_amount = 5
options.be_verbose = True
options.num_threads = 4
# =============================================================================



# PREPROCESS DATA
# =============================================================================
print('preprocessing data...')

# get left eye landmarks subset of the ibug annotations
save_eye_landmarks(ibug_data_path, eye_data_path)
# =============================================================================



# TRAIN MODEL
# =============================================================================
print('training shape predictor...')
dlib.train_shape_predictor(eye_data_path, eye_predictor_path, options)
# =============================================================================