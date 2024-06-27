import cv2
import numpy as np
import os


output_dir = 'augmented_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = cv2.imread(r'C:\Users\Personal\Documents\projects\KICS Second Project\augmentation\cats.jpg', 1)


if img is None:
    print("Error: Unable to load image.")
    exit()




def flip_horizontal(image):
    flipped_image = cv2.flip(image, 1)
    print("Flipping horizontally")
    return flipped_image

def flip_vertical(image):
    flipped_ver = cv2.flip(image, 0)
    print("Flipping horizontally")
    return flipped_ver


def rotate_90(image):
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    print("Rotating 90 degrees")
    return rotated_image


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



# Original
cv2.imwrite(os.path.join(output_dir, 'original.png'), img)


flipped_img = flip_horizontal(img)
cv2.imwrite(os.path.join(output_dir, 'flipped_horizontal.png'), flipped_img)
flipped_img1 = flip_vertical(img)
cv2.imwrite(os.path.join(output_dir, 'flipped_vertical.png'), flipped_img1)


rotated_img = rotate_90(img)
cv2.imwrite(os.path.join(output_dir, 'rotated_90.png'), rotated_img)


brighter_img = adjust_brightness(img, 50)
cv2.imwrite(os.path.join(output_dir, 'brighter.png'), brighter_img)


blurred_img = gaussian_blur(img)
cv2.imwrite(os.path.join(output_dir, 'gaussian_blur.png'), blurred_img)

print("Image augmentation complete. Augmented images saved to 'augmented_images' directory.")
