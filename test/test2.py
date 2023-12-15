import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelBinarizer

physical_device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0],True)

#LOAD DATA
(x_train, y_train),(x_test,y_test) = tfds.load(
    name="emnist/byclass",
    split=['train','test'],
    as_supervised=True,
    shuffle_files=True,
    batch_size=-1
)
#PREPROCESSING DATA
x_train = tfds.as_numpy(x_train)
y_train = tfds.as_numpy(y_train)
x_test = tfds.as_numpy(x_test)
y_test = tfds.as_numpy(y_test)
input_shape = (28,28,1)

#NORMALISE DATA
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# print(x_train.shape, y_train.shape)
# print(y_train[0:20])

#CLASS WEIGHT ASSIGNMENT FOR IMBALANCED DATASET
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
# print(y_train[0:20])
counts = y_train.sum(axis=0)
# print(counts)
classTotals = y_train.sum(axis=0)
classWeight = {}

for i in range(0,len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

#MODEL BUILDING RESNET
BATCH_SIZE = 128
EPOCH = 2
lr = 0.001
model = tf.keras.models.Sequential(
    [
        keras.layers.InputLayer((28,28,1)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64,3,activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128,3,activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        # keras.layers.Dense(1024, activation='relu'),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(62, activation='softmax')
    ]
)

# MODEL COMPILE
# Multiple options of optimizer and loss funciton
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy']
)
# MODEL FIT
model.fit(x_train,y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, class_weight=classWeight)

# MODEL EVALUATION
model.evaluate(x_test,y_test)
