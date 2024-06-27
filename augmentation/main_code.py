import cv2
import numpy as np
import os

from specifications import flip_horizontal, flip_vertical, rotate_90, brightness_level, gaussian_blur

output_dir = 'augmented_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread('cats.jpg', 1)
