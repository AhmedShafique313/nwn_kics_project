import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define paths to training and testing data folders
train_data_path = r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\dataset'
test_data_path = r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\valid'

# Load mel spectrogram plots
mel_spectrograms = []
labels = []

# Load mel spectrogram plots and their corresponding labels
# For training data
for file in os.listdir(train_data_path):
    if file.endswith('.wav'):
        # Assuming your training data files are named with their class label 
        # Example: 'cat_1.wav', 'dog_2.wav'
        label = file.split('_')[0]
        file_path = os.path.join(train_data_path, file)
        mel_spectrogram, sr = librosa.load(file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=mel_spectrogram, sr=sr, n_mels=40)
        mel_spectrograms.append(mel_spectrogram)
        labels.append(label)

# For testing data
for file in os.listdir(test_data_path):
    if file.endswith('.wav'):
        # Assuming your testing data files are named with their class label 
        # Example: 'cat_1.wav', 'dog_2.wav'
        label = file.split('_')[0]
        file_path = os.path.join(test_data_path, file)
        mel_spectrogram, sr = librosa.load(file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=mel_spectrogram, sr=sr, n_mels=40)
        mel_spectrograms.append(mel_spectrogram)
        labels.append(label)

# Convert mel spectrograms and labels to numpy arrays
mel_spectrograms = np.array(mel_spectrograms)
labels = np.array(labels)

# One-hot encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mel_spectrograms, labels, test_size=0.2, random_state=42)

# Define RNN model
model = keras.Sequential([
    keras.layers.LSTM(units=128, return_sequences=True, input_shape=(mel_spectrograms.shape[1], mel_spectrograms.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=64),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=8, activation='softmax')
])

# Compile RNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train RNN model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate RNN model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

# Use RNN model to make predictions
predictions = model.predict(X_test)