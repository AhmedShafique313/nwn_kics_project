import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile

dirr = '/home/ali/Desktop/GDS/Data/card data/c/'   # set path of the .wav file directory
output_dir = '/home/ali/Desktop/GDS/Data/card data/a/'  # set path of the output directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lst = os.listdir(dirr)
for f in lst:
    if f.endswith('.wav'):
        filepath = os.path.join(dirr, f)
        sample_rate, audio_data = wavfile.read(filepath)

        # Find the indices of the chunks that meet the threshold condition
        chunk_indices = np.where(audio_data > 180)[0]
        for i in range(len(chunk_indices) - 1):
            start_idx = chunk_indices[i]
            end_idx = chunk_indices[i + 1]

            # Create a plot for the current chunk
            plt.plot(audio_data[start_idx:end_idx])
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.title(f + ' Chunk ' + str(i))
            plt.savefig(os.path.join(output_dir, f + 'chunk' + str(i) + '.png'))
            plt.clf()