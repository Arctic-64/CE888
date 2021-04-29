import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import CSVLogger
import pathlib

from keras.callbacks import History
import csv

import pandas as pd

df = pd.read_csv (r'training_log')
int (df)

y = df["accuracy"]
x = range(len(df))

plt.plot(x,y, label='training_set')

y = df["val_accuracy"]
x = range(len(df))

plt.plot(x,y, label='validation_set')

plt.xlabel('Epochs')
plt.ylabel('Model_accuracy')

plt.title("Accuracy over time")
plt.legend()
plt.show()


y = df["loss"]
x = range(len(df))

plt.plot(x,y, label='training_set')

y = df["val_loss"]
x = range(len(df))

plt.plot(x,y, label='validation_set')

plt.xlabel('Epochs')
plt.ylabel('Model_loss')

plt.title("loss over time")
plt.legend()
plt.show()
