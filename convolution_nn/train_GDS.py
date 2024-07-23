import glob
import os
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
from sklearn.utils import shuffle

def get_image_data(image_path):
    im = img_to_array(load_img(image_path).resize((256, 256), Image.LANCZOS))
    im = im / 255.0
    return im

def data(path, label):
    X = []
    Y = []
    image_paths = []
    files = glob.glob(path + '/*.png')
    for file in files:
        image = get_image_data(file)
        X.append(image)
        Y.append(label)
        image_paths.append(file)
    return X, Y, image_paths

# Dimensions of images:
img_width, img_height = 256, 256

# Data directories:
train_data_dir_gun = r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\dataset'
validation_data_dir_gun = r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\valid'

# Hyperparameters:
epochs = 10
batch_size = 25

# Load training data:
train_x1, train_y1, path_train1 = data(train_data_dir_gun, 1)
train_x = train_x1
train_y = train_y1
train_x, train_y = shuffle(train_x, train_y)
train_x = np.array(train_x)
train_y = np.array(train_y)
train_y = to_categorical(train_y, 2)

# Load validation data:
val_x1, val_y1, path_val_1 = data(validation_data_dir_gun, 1)
val_x = val_x1
val_y = val_y1
val_x, val_y = shuffle(val_x, val_y)
val_x = np.array(val_x)
val_y = np.array(val_y)
val_y = to_categorical(val_y, 2)

# Build the model:
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model:
model.fit(train_x, train_y, epochs=15, batch_size=25, validation_data=(val_x, val_y))

# Save the model:
model.save(r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\new_melspecto.keras')

# Test prediction:
gunshot = []
true_gun = []

for im in val_x1:
    im = np.array(im)
    im = np.reshape(im, newshape=(1, 256, 256, 3))
    prediction = model.predict(im)
    classes_x = np.argmax(prediction, axis=1)

    gunshot.append(classes_x)
    true_gun.append(1)
    print("True class", 1, " predicted class", classes_x)

final_arrays = [path_val_1, true_gun, gunshot]

# Save results to CSV without using zip:
with open(r'C:\Users\Personal\Documents\projects\KICS Second Project\convolution_nn\test.csv', "a") as f:
    for i in range(len(final_arrays[0])):
        f.write(f"{final_arrays[0][i]},{final_arrays[1][i]},{final_arrays[2][i][0]}\n")
