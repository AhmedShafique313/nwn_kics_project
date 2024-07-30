import Augmentor
from PIL import Image
import os
import matplotlib.pyplot as plt

# Function to load and display an image
def load_and_display_image(file_path):
    img = Image.open(file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

# Path to the directory containing your Mel spectrogram images
input_dir = r"C:\Users\Personal\Documents\projects\KICS Second Project\mel_augmentation\input"
output_dir = r"C:\Users\Personal\Documents\projects\KICS Second Project\mel_augmentation\output"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize the Augmentor pipeline
p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)

# Add augmentation operations
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)
p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
p.random_color(probability=0.5, min_factor=0.7, max_factor=1.3)
p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)

# Sample from the pipeline
p.sample(20)  

# Display one of the augmented images
augmented_image_path = os.path.join(output_dir, os.listdir(output_dir)[0])
augmented_image = load_and_display_image(augmented_image_path)
