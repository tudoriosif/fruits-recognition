import numpy as np
import os, os.path
from sklearn.datasets import load_files
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import load_model

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


# Get test data
train_dir = './fruits-360/Training'
test_dir = './fruits-360/Test'
_, _, target_labels = load_dataset(train_dir)
x_test_init, y_test_init, _ = load_dataset(test_dir)

# Reduce test data
x_test_init = x_test_init[7000:]
y_test_init = y_test_init[7000:]

# Process test data
x_test = np.array(convert_image_to_array_form(x_test_init))
y_test = to_categorical(y_test_init, 131)

# Load existing model
model = load_model('fruit-classifier.keras')

# Predict
predictions = model.predict(x_test)
fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(mpimg.imread(x_test_init[idx]))
    pred_idx = np.argmax(predictions[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} / Pred: {}".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
plt.show()
