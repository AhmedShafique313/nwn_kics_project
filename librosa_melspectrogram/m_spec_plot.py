import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the audio file
audio_path = r'C:\\Users\\Personal\\Documents\\projects\\KICS Second Project\\librosa_melspectrogram\\valid 3.wav'

# Load the audio file
y, sr = librosa.load(audio_path, sr=None)


sample_range = 12000
y = y[:sample_range]
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert the power spectrogram (amplitude squared) to decibel (dB) units
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot the mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (First 12,000 samples)')
plt.tight_layout()

# Ensure the output directory exists
output_directory = r'C:\\Users\\Personal\\Documents\\projects\\KICS Second Project\\librosa_melspectrogram\\output_directory\\'
os.makedirs(output_directory, exist_ok=True)

# Path to save the output image
output_image_path = os.path.join(output_directory, 'valid3_mel.png')

# Save the plot as a PNG file
plt.savefig(output_image_path)
plt.show()

print(f"Spectrogram saved to {output_image_path}")
