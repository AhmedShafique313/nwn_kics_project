import cv2
import numpy as np
import os

from specifications import flip_horizontal, flip_vertical, rotate_90, adjust_brightness, gaussian_blur

output_dir = 'augmented_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread(r'C:\Users\Personal\Documents\projects\KICS Second Project\augmentation\cats.jpg', 1)

# calling all stuff in main file
cv2.imwrite(os.path.join(output_dir, 'original.png'), img)

flipped_h_img = flip_vertical(img)
cv2.imwrite(os.path.join(output_dir, 'horizontal flipped.png'), img)

flipped_v_img = flip_vertical(img)
cv2.imwrite(os.path.join(output_dir, 'vertically fliiped.png'), img)

rotated = rotate_90(img)
cv2.imwrite(os.path.join(output_dir, 'rotated image.png'), img)

brighted = adjust_brightness(img, 50)
cv2.imwrite(os.path.join(output_dir, 'brighted image.png'), img)

blured = gaussian_blur(img)
cv2.imwrite(os.path.join(output_dir, 'blured image.png'), img)

print('Image augmentation complete and all the augmented stuff stored in augmentation_dataset folder')
