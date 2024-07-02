import cv2
import numpy as np
import os
from scipy.io.wavfile import write

image_path = r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\output_audio\youtube_shot.wav.png'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
amplitude_data = gray_image.mean(axis=1)


amplitude_data = (amplitude_data - np.mean(amplitude_data)) / np.max(np.abs(amplitude_data))
amplitude_data = np.int16(amplitude_data * 32767)

# Original file sample rate is 44100
sample_rate = 44100  

output_dir = r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\audio_output'
output_path = os.path.join(output_dir, 'reconstructed_audio.wav')
write(output_path, sample_rate, amplitude_data)

print(f'Audio file saved at: {output_path}')
