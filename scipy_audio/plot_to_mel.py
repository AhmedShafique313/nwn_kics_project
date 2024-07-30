import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image, ImageEnhance, ImageOps
import os

# Load audio file
audio_file = 'path/to/your/audio/file.wav'
audio_data, sample_rate = librosa.load(audio_file)

# Define sample range
start_sample = 0  # Starting point of the sample range
end_sample = 12000  # Ending point of the sample range

# Generate Mel spectrogram for the specified sample range
mel_spectrogram = librosa.feature.melspectrogram(y=audio_data[start_sample:end_sample], sr=sample_rate, n_mels=128)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Save the Mel spectrogram as an image
output_dir = "path/to/save/mel_spectrogram_images"
os.makedirs(output_dir, exist_ok=True)
mel_spectrogram_path = os.path.join(output_dir, 'mel_spectrogram.png')

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel spectrogram (Samples {start_sample} to {end_sample})')
plt.tight_layout()
plt.savefig(mel_spectrogram_path)
plt.close()

# Load and display the saved Mel spectrogram image
def load_and_display_image(file_path):
    img = Image.open(file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

# Display the original Mel spectrogram image
original_image = load_and_display_image(mel_spectrogram_path)

# Define a function for augmentation
def augment_image(image):
    augmented_images = []
    
    # Rotate image
    rotated_image = image.rotate(25)
    augmented_images.append(rotated_image)
    
    # Flip image horizontally
    flipped_lr_image = ImageOps.mirror(image)
    augmented_images.append(flipped_lr_image)
    
    # Flip image vertically
    flipped_tb_image = ImageOps.flip(image)
    augmented_images.append(flipped_tb_image)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(1.5)  # Increase brightness by 50%
    augmented_images.append(brightened_image)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    contrasted_image = enhancer.enhance(1.5)  # Increase contrast by 50%
    augmented_images.append(contrasted_image)
    
    return augmented_images

# Perform augmentations
augmented_images = augment_image(original_image)

# Save and display augmented images
for i, aug_img in enumerate(augmented_images):
    aug_img_path = os.path.join(output_dir, f'augmented_image_{i}.png')
    aug_img.save(aug_img_path)
    load_and_display_image(aug_img_path)

print("Original and augmented images saved and displayed successfully!")
