# import numpy as np
# from scipy.io import wavfile
# import matplotlib.pyplot as plt

# # Load the audio data from files
# sample_rate1, audio1 = wavfile.read(r'C:\Users\Personal\Documents\projects\KICS Second Project\librosa_melspectrogram\S4.wav')
# sample_rate2, audio2 = wavfile.read(r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\valid12.wav')

# # Ensure both audio files have the same sample rate
# if sample_rate1 != sample_rate2:
#     raise ValueError("Sample rates of the two audio files do not match.")

# # Convert audio data to mono if necessary
# def convert_to_mono(data):
#     if len(data.shape) == 2:
#         data = np.mean(data, axis=1)
#     return data

# audio1 = convert_to_mono(audio1)
# audio2 = convert_to_mono(audio2)

# # Normalize the audio data
# def normalize_audio(data):
#     return data / np.max(np.abs(data))

# audio1 = normalize_audio(audio1)
# audio2 = normalize_audio(audio2)

# # Plot the individual audio signals
# def plot_audio_signals(audio1, audio2, sample_rate):
#     time1 = np.linspace(0, len(audio1) / sample_rate, num=len(audio1))
#     time2 = np.linspace(0, len(audio2) / sample_rate, num=len(audio2))
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(time1, audio1, label='Local Audio sample', color='blue')
#     plt.plot(time2, audio2, label='Internet audio sample', color='green')
#     plt.title('Gunshot Audio Signals')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_audio_signals(audio1, audio2, sample_rate1)

# # Save the audio data to .wav files
# wavfile.write('local_audio_sample.wav', sample_rate1, audio1)
# wavfile.write('internet_audio_sample.wav', sample_rate2, audio2)

# print("Audio files saved successfully!")


import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Load the audio data from files
sample_rate1, audio1 = wavfile.read(r'C:\Users\Personal\Documents\projects\KICS Second Project\librosa_melspectrogram\S4.wav')
sample_rate2, audio2 = wavfile.read(r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\valid16.wav')

# Ensure both audio files have the same sample rate
if sample_rate1 != sample_rate2:
    raise ValueError("Sample rates of the two audio files do not match.")

# Convert audio data to mono if necessary
def convert_to_mono(data):
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    return data

audio1 = convert_to_mono(audio1)
audio2 = convert_to_mono(audio2)

# Normalize the audio data
def normalize_audio(data):
    return data / np.max(np.abs(data))

audio1 = normalize_audio(audio1)
audio2 = normalize_audio(audio2)

# Convert normalized audio back to int16 for .wav saving
audio1_int16 = (audio1 * 32767).astype(np.int16)
audio2_int16 = (audio2 * 32767).astype(np.int16)

# Plot the individual audio signals
def plot_audio_signals(audio1, audio2, sample_rate):
    time1 = np.linspace(0, len(audio1) / sample_rate, num=len(audio1))
    time2 = np.linspace(0, len(audio2) / sample_rate, num=len(audio2))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time1, audio1, label='Local Audio sample', color='blue')
    plt.plot(time2, audio2, label='Internet audio sample', color='green')
    plt.title('Gunshot Audio Signals')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_audio_signals(audio1, audio2, sample_rate1)

# Save the audio data to .wav files
wavfile.write('local_audio_sample.wav', sample_rate1, audio1_int16)
wavfile.write('internet_audio_sample16.wav', sample_rate2, audio2_int16)

print("Audio files saved successfully!")
