import cv2
import numpy as np
import os

from specifications import flip_horizontal, flip_vertical, rotate_90, brightness_level, gaussian_blur

output_dir = 'augmented_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread('cats.jpg', 1)

# calling all stuff in main file
cv2.imwrite(os.path.join(output_dir, 'original.png'), img)

flipped_h_img = flip_vertical(img)
cv2.imwrite(os.path.join(output_dir, 'horizontal flipped'), img)

flipped_v_img = flip_vertical(img)
cv2.imwrite(os.path.join(output_dir, 'vertically fliiped'), img)

rotated = rotate_90(img)
cv2.imwrite(os.path.join(output_dir, 'rotated image'), img)

