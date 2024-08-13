from PIL import Image
import os

# Set the desired image size
desired_size = (256, 256)

# Set the path to the images directory
images_dir = r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\images valid'

# Set the path to the resized images directory
resized_images_dir = r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\resized_images'

# Create the resized images directory if it doesn't exist
if not os.path.exists(resized_images_dir):
    os.makedirs(resized_images_dir)

# Loop through all images in the directory
for filename in os.listdir(images_dir):
    # Open the image file
    img = Image.open(os.path.join(images_dir, filename))
    
    # Resize the image
    img = img.resize(desired_size)
    
    # Save the resized image
    img.save(os.path.join(resized_images_dir, filename))