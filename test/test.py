# import tensorflow as tf
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64)

# train_new_model = True

# if train_new_model:
#     # Loading the MNIST data set with samples and splitting it
#     mnist = tf.keras.datasets.mnist
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()

#     # Normalizing the data (making length = 1)
#     X_train = tf.keras.utils.normalize(X_train, axis=1)
#     X_test = tf.keras.utils.normalize(X_test, axis=1)

#     # Create a neural network model
#     # Add one flattened input layer for the pixels
#     # Add two dense hidden layers
#     # Add one dense output layer for the 10 digits
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#     model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#     model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

#     # Compiling and optimizing model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Training the model
#     model.fit(X_train, y_train, epochs=3)

#     # Evaluating the model
#     val_loss, val_acc = model.evaluate(X_test, y_test)
#     print(val_loss)
#     print(val_acc)

#     # Saving the model
#     model.save('handwritten_digits.model')
# else:
#     # Load the model
#     model = tf.keras.models.load_model('handwritten_digits.model')

#Import libraries 

from multiprocessing import Condition
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#Data Load 

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)


BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)


BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

#Model 

input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #hidden layer 1 
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'), #hidden layer 2 
                            tf.keras.layers.Dense(output_size, activation='softmax')   
                            ])

#Optimiser and Loss function

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training

NUM_EPOCHS = 5

model.fit(train_data, epochs = NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=1)

#Test the model 

test_loss, test_accuracy = model.evaluate(test_data)

print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))