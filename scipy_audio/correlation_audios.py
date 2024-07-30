import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate

def load_audio(file_path):
    # Load the audio file
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def convert_to_mono(data):
    # If the audio is stereo (2 channels), convert it to mono
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    return data

def normalize_audio(data):
    # Normalize audio data to range [-1, 1]
    return data / np.max(np.abs(data))

def plot_audio_signals(audio1, audio2, sample_rate):
    # Plot the audio signals
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

def plot_correlation(audio1, audio2, sample_rate):
    # Compute the correlation of the two audio signals
    correlation = correlate(audio1, audio2, mode='full')
    lags = np.arange(-len(audio1) + 1, len(audio2))
    
    # Separate the correlation based on the audio signals
    half = len(correlation) // 2
    correlation_audio1 = correlation[:half]
    correlation_audio2 = correlation[half:]
    
    # Plot the correlation with different labels
    plt.figure(figsize=(12, 6))
    plt.plot(lags[:half] / sample_rate, correlation_audio1, label='Local Audio sample', color='blue')
    plt.plot(lags[half:] / sample_rate, correlation_audio2, label='Internet Audio sample', color='green')
    plt.title('Correlation between two gunshot audio signals')
    plt.xlabel('Time lag (seconds)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load the two gunshot audio files
file_path1 = r'C:\Users\Personal\Documents\projects\KICS Second Project\librosa_melspectrogram\S4.wav'
file_path2 = r'C:\Users\Personal\Documents\projects\KICS Second Project\scipy_audio\valid13.wav'
sample_rate1, audio1 = load_audio(file_path1)
sample_rate2, audio2 = load_audio(file_path2)

# Ensure both audio files have the same sample rate
if sample_rate1 != sample_rate2:
    raise ValueError("Sample rates of the two audio files do not match.")

# Convert audio data to mono if necessary
audio1 = convert_to_mono(audio1)
audio2 = convert_to_mono(audio2)

# Normalize the audio data
audio1 = normalize_audio(audio1)
audio2 = normalize_audio(audio2)

# Plot the individual audio signals
plot_audio_signals(audio1, audio2, sample_rate1)

# Plot the correlation
plot_correlation(audio1, audio2, sample_rate1)

