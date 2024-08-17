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
train_data_path = r'C:\Users\Personal\Documents\projects\KICS Second Project\RNN_model\train'
test_data_path = r'C:\Users\Personal\Documents\projects\KICS Second Project\RNN_model\test'

# Define number of Mel frequency bins
n_mels = 40

# Define a function to extract Mel spectrograms
def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return mel_spectrogram

# Load mel spectrogram plots and their corresponding labels
# For training data
X_train = []
y_train = []
for file in os.listdir(train_data_path):
    if file.endswith('.wav'):
        # Assuming your training data files are named with their class label 
        # Example: 'cat_1.wav', 'dog_2.wav'
        label = file.split('_')[0]
        file_path = os.path.join(train_data_path, file)
        mel_spectrogram = extract_mel_spectrogram(file_path)
        X_train.append(mel_spectrogram)
        y_train.append(label)

# For testing data
X_test = []
y_test = []
for file in os.listdir(test_data_path):
    if file.endswith('.wav'):
        # Assuming your testing data files are named with their class label 
        # Example: 'cat_1.wav', 'dog_2.wav'
        label = file.split('_')[0]
        file_path = os.path.join(test_data_path, file)
        mel_spectrogram = extract_mel_spectrogram(file_path)
        X_test.append(mel_spectrogram)
        y_test.append(label)

# Convert mel spectrograms and labels to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# One-hot encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = le.transform(y_test)
y_test = tf.keras.utils.to_categorical(y_test)

# Define RNN model
model = keras.Sequential([
    keras.layers.LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=64),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=y_train.shape[1], activation='softmax')
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