import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile

# Set the paths of the input and output directories
dirr = r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\Sniper rifle audio dataset'
output_dir = r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\output_audio/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lst = os.listdir(dirr)

for f in lst:
    if f.endswith('.wav'):
        filepath = os.path.join(dirr, f)
        sample_rate, audio_data = wavfile.read(filepath)

        # Plot the entire audio data
        plt.plot(audio_data)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title(f)
        plt.savefig(os.path.join(output_dir, f + '.png'))
        plt.clf()

        print("Image created for file:", f)
        break  # Exit the loop after processing the first file
