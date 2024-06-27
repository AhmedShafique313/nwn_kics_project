import os
import cv2
import numpy as np

def flip_horizontal(image):
    flipped_image = cv2.flip(image, 1)
    print("Flipping horizontally")
    return flipped_image

def rotate_90(image):
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    print("Rotating 90 degrees")
    return rotated_image

def flip_vertical(image):
    flipped_ver = cv2.flip(image, 0)
    print("Flipping Vertically")
    return flipped_ver

def adjust_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    brighter_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    print("Adjusting brightness")
    return brighter_image

def gaussian_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    print("Applying Gaussian blur")
    return blurred_image
