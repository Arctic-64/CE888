import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import CSVLogger
import pathlib
from keras.utils.vis_utils import plot_model
from keras.callbacks import History
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

load_model = True
training_data = "Training"
testing_data = "Test"
epochs = 20
csv_logger = CSVLogger("training_log", separator=",", append=True)


batch_size = 50
image_size = (256, 256)

# Get all the paths
data_dir_list = os.listdir('data')
#print(data_dir_list)
path, dirs, files = next(os.walk("data"))
file_count = len(files)
print(file_count)

IMG_SIZE = 224

train_dir = "data/training"
validation_dir = "data/val"
test_dir = "data/test"

# Make new base directory
original_dataset_dir = 'data/comp'
base_dir = 'data/base'


train_dir = os.path.join(base_dir, 'train')


validation_dir = os.path.join(base_dir, 'validation')


test_dir = os.path.join(base_dir, 'test')


import shutil

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dir, fname)
    # print(src,dst)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dir, fname)
    shutil.copyfile(src, dst)

import cv2
import numpy as np
from random import shuffle

IMG_SIZE = 224
LR = 1e-4
def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'dog': return 1
    elif word_label == 'cat': return 0

def createDataSplitSet(datapath):
    X=[]
    y=[]

    for img in os.listdir(datapath):
        label = label_img(img)
        # print(label)
        path = os.path.join(datapath, img)
        image = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE))
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        X.append(np.array(image))
        y.append(label)

    return np.array(X), np.array(y)

train_X, train_y = createDataSplitSet(train_dir)
val_X, val_y = createDataSplitSet(validation_dir)
test_X, test_y = createDataSplitSet(test_dir)




if load_model:
    model = VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=img_input,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax")
    model.summary()

    last_layer = model.get_layer('fc2').output
    out = Dense(1, activation='sigmoid', name='output')(last_layer)  ## 2 classes
    model = Model(img_input, out)

    for layer in model.layers[:-1]:
        layer.trainable = False

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    my_callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath='vgg16_model.h5', save_best_only=True),
    ]

def mode_train():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_data,
        validation_split=0.2,
        subset="training",
        seed=9550,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_data,
        validation_split=0.2,
        subset="validation",
        seed=9550,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[csv_logger]
    )
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("MAE for binary classification")
    plt.xlabel("No. epoch")
    plt.ylabel("MAE value")
    plt.show()
    model.save(model_savename)


def mode_test():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        testing_data,
        batch_size=50,
        image_size=image_size,
        shuffle=True,
        seed=9550,
    )
    results = model.evaluate(dataset, batch_size=batch_size)
    print("test loss, test acc:", results)
    percentage = results[1]
    percentage = percentage * 100
    print("percentage accuracy = ", percentage)


history = model.fit(train_X, train_y,
                               batch_size=10,
                               epochs=10,
                               validation_data=(val_X, val_y),
                               callbacks=my_callbacks)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import accuracy_score

## Test Accuracy
predictions = model.predict(test_X)
ypred = predictions > 0.5
test_acc = accuracy_score(test_y, ypred)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

precision, recall, f1score, _ = precision_recall_fscore_support(test_y, ypred, average='binary')

auc = roc_auc_score(test_y, ypred)

print("Train Accuracy:\t", acc[-1])
print("Val Accuracy:\t", val_acc[-1])
print("Test Accuracy:\t", test_acc)
print("Precision:\t", precision)
print("Recall:\t\t", recall)
print("F1 Score:\t", f1score)
print("AUC:\t\t", auc)

