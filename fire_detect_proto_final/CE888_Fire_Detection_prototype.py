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
from keras.layers import experimental
history = History()

load_model = False
model_name = "Fire_detect_prototype_V1"
model_savename = "Fire_detect_prototype_V1"
mode = "generate"
#modes, "generate", "test", "train"
### WARNING VERY POSSIBLE OVERWRITE WHEN RUNNING,
###DOUBLE AND TRIPPLE CHECK THE MODE AND FILE NAME VARIABLES
training_data = "Training"
testing_data = "Test"
epochs = 20
csv_logger = CSVLogger("training_log", separator=",", append=True)


batch_size = 50
image_size = (256, 256)

if load_model:
    model = tf.keras.models.load_model(model_name)
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

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


def mode_generate():
    inputs = keras.Input(shape=image_size)
    x = inputs

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    # for size in [128, 256, 512, 728]:
    for size in [8]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)

        x = layers.add([x, residual])
        previous_block_activation = x
    x = layers.SeparableConv2D(8, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="model_fire")




if mode == 'test':
    mode_test()
elif mode == 'train':
    mode_train()
elif mode == 'generate':
    if load_model:
        print("model is set to load")
        quit(1)
    model = mode_generate()
    model.save(model_savename)
else:
    print("unrecognised mode")