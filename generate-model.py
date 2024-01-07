import numpy as np
import os, os.path
from sklearn.datasets import load_files
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Data loading and preprocessing functions
def load_dataset(data_path):
    data_loading = load_files(data_path)
    files_add = np.array(data_loading['filenames'])
    targets_fruits = np.array(data_loading['target'])
    target_labels_fruits = np.array(data_loading['target_names'])
    return files_add, targets_fruits, target_labels_fruits

def convert_image_to_array_form(files):
    images_array = []
    for file in files:
        images_array.append(img_to_array(load_img(file)))
    return np.array(images_array)

# Neural network model function
def tensorflow_based_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(100, 100, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(131, activation='softmax'))
    return model

# Load datasets
train_dir = './fruits-360/Training'
test_dir = './fruits-360/Test'
x_train, y_train, target_labels = load_dataset(train_dir)
x_test, y_test, _ = load_dataset(test_dir)
x_test, x_valid = x_test[7000:], x_test[:7000]
y_test, y_valid = y_test[7000:], y_test[:7000]

# Preprocess datasets
x_train = np.array(convert_image_to_array_form(x_train))
x_test = np.array(convert_image_to_array_form(x_test))
x_valid = np.array(convert_image_to_array_form(x_valid))
y_train = to_categorical(y_train, 131)
y_test = to_categorical(y_test, 131)
y_valid = to_categorical(y_valid, 131)


# Build and compile the model
model = tensorflow_based_model()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_valid, y_valid),
    verbose=2, shuffle=True
)

# # Save the model
model.save('fruit-classifier.keras')

# # Output the model summary
model.summary()

# Test model
acc_score = model.evaluate(x_test, y_test) #we are starting to test the model here
print('\n', 'Test accuracy:', acc_score[1])