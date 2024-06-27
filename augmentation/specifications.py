import os
import cv2
import numpy as np

def flip_horizontal(image):
    return cv2.flip(image, 1)

def rotate_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def flip_vertical(image):
    return cv2.flip(image, 0)

def brightness_level(image, value=30):
    # converting it to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v= cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

