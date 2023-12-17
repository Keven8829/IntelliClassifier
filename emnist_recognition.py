import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow import keras
import tensorflow_datasets as tfds

# If you use GPU this might save you some error
physical_device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

# Configuration to build model
build_model = True

# load data from tensorflow dataset 
dataset,ds_info = tfds.load(
    name="emnist/byclass",
    split=['train[:90%]','train[90%:]','test'],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
)

ds_train = dataset[0]
val_train = dataset[1]
ds_test = dataset[2]

# PREPROCESSING DATA
def normalize_img(image, label):
    """Normalizes images"""
    # Rotate 90 degrees anticlockwise
    image = tf.transpose(image, perm=[1, 0, 2])
    return (tf.cast(image, tf.float32) / 255.0), label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
EPOCH = 20

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train[:90%]"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for validation dataset
val_train = val_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
val_train = val_train.cache()
val_train = val_train.shuffle(ds_info.splits["train[90%:]"].num_examples)
val_train = val_train.batch(BATCH_SIZE)
val_train = val_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)


if build_model:
    # Build CNN Model
    model = tf.keras.models.Sequential(
        [
            # keras.layers.InputLayer((28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
            keras.layers.BatchNormalization(),
            # keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(32,3,activation='relu'),
            keras.layers.BatchNormalization(),
            # keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(32,5,activation='relu',padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Conv2D(64,3,activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64,3,activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64,5,activation='relu',padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Conv2D(128,4,activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            # Dense(256,activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(62, activation="softmax"),

            
        ]
    )
    # print(model.summary())


    # COMPILING MODEL
    """{'RMSDrop', 'Adam', 'Adamax', 'SGD', 'Adadelta'}"""
    
    # Get user input for optimizer choice
    optimizer_train = input("Choose optimizer: {'RMSprop', 'Adam', 'Adamax', 'SGD', 'Adadelta'} ")

    # Map user input to the corresponding optimizer instance
    if optimizer_train == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    elif optimizer_train == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer_train == 'Adamax':
        optimizer = keras.optimizers.Adamax(learning_rate=0.001)
    elif optimizer_train == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=0.001)
    elif optimizer_train == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(learning_rate=1.0)
    else:
        raise ValueError("Invalid optimizer train")

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Model fit
    callback = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.005)
    ]
    val_x, val_y = next(iter(val_train))
    history = model.fit(ds_train, epochs=EPOCH, verbose=1, batch_size=BATCH_SIZE, callbacks=callback, validation_data=(val_x,val_y))

    # Model Evaluation
    model.evaluate(ds_test)

    # Save the model with optimizer in the filename
    model.save(f"model/emnist_{optimizer_train.lower()}.h5")
    print(f"Saving the model as emnist_{optimizer_train.lower()}.h5")

    # Save graph dan data
    history_dict = history.history # Extract training history
    history_df = pd.DataFrame(history_dict) # Convert to DataFrame
    history_df.to_csv(f'data/training_history_{optimizer_train.lower()}.csv', index=False)# Save to CSV file

    #show graph
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid()
    plt.show()


"""-----------------------------------------------------------------------------------------------"""

# Class Label
label_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Predict from split dataset to test the model before predict
def predict_from_sample():
    loaded_model = tf.keras.models.load_model("model/emnist.h5")
    # Get one batch from the test dataset
    for images, labels in ds_test.take(1):
        # Predict using the loaded model
        predictions = loaded_model.predict(images)

        # Access individual prediction and label if needed
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = labels[i].numpy()

            # Print or use prediction and label as needed
            print(f"Predicted Label: {np.argmax(prediction)}, True Label: {label}")

        # Visualize an example image from the batch
        example_image = images[4].numpy().reshape(28, 28)
        plt.imshow(example_image, cmap='gray')
        # plt.title("{}".format())
        plt.colorbar()
        plt.show()


def load_and_predict_image(image_path, optimizer_choice, top_n = 3):
    loaded_model = tf.keras.models.load_model(f"model/emnist_{optimizer_choice.lower()}.h5")
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to 28x28 (assuming it's a square image)
    resized_image = cv2.resize(image, (28, 28))
    
    # Invert black and white
    inverted_image = 255 - resized_image

    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = inverted_image / 255.0

    # Add batch dimension and channel dimension to match the model input shape
    input_image = np.expand_dims(np.expand_dims(normalized_image, axis=-1), axis=0)
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_image)

    # Display the top predictions
    top_classes = np.argsort(prediction.ravel())[-top_n:][::-1]
    top_probabilities = prediction.ravel()[top_classes]

    label_list = [];
    for label in top_classes:
        label_list.append(label_names[label])
    return label_list, top_probabilities


def show_stats(optimizer_choice):
    # Read the CSV file into a DataFrame
    history_df = pd.read_csv(f'training_history_{optimizer_choice.lower()}.csv')

    # Plot training and validation loss
    fig_loss = plt.figure(figsize=(10, 6))
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # Plot training and validation accuracy
    fig_accuracy = plt.figure(figsize=(10, 6))
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    return fig_loss , fig_accuracy

