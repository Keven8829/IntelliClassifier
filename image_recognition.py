import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from keras import models
from keras import layers
from keras import datasets

class ImageRecognitionModel:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.train_class = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
        self.statistic_class = ['adam_history.csv', 'adadelta_history.csv', 'adagrad_history.csv']
        self.optimizer_class = ['adam', 'adadelta', 'adagrad']
        self.model_class = ['Cifar100.model50', 'Cifar100.model50a', 'Cifar100.model50b']
    
    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(self.img_height, self.img_width, 3), padding='same')) #add padding , add more convo
        self.model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
        self.model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Conv2D(128, (3, 3), activation=tf.nn.relu))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation=tf.nn.relu))
        self.model.add(layers.Dense(100, activation=tf.nn.softmax))

    def train_model(self, train_images, train_labels): #add evalutate
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(train_images, train_labels, epochs=50)
        return history
    
    def plot_statistic(self, history, save_csv_path='adam_history.csv'): #if want to plot this history statistic then save is as csv
        if save_csv_path:
            df = pd.DataFrame(history.history)
            df.index.name = 'Epoch'
            df.to_csv(save_csv_path)

    def statistic_history_data(self, statistic_file_index):
        df = pd.read_csv(self.statistic_class[statistic_file_index])
        plt.figure(figsize=(10, 6))
        plt.plot(df['accuracy'])
        plt.plot(df['loss'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Train')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.show()

    def save_model(self, filename="Cifar100.model50"):
        if self.model:
            self.model.save(filename)
            print(f"Model saved as {filename}")
        else:
            print("Cannot save the model. Model not initialized.")

    def load_model(self, filename_index):
        try:
            self.model = models.load_model(self.model_class[filename_index])
            print(f"Model loaded from {self.model_class[filename_index]}")
        except Exception as e:
            print(f"Error loading model from {self.model_class[filename_index]}: {e}")

    def model_evaluate(self):
        loss, accuracy = self.model.evaluate(test_images,test_labels)
        print(loss)
        print(accuracy)

    def model_function(self):
        if not self.model:
            self.build_model()
            history = self.train_model(train_images, train_labels)
            self.plot_statistic(history)
            self.model.evaluate(test_images,test_labels)
            self.save_model()
        else:
            print("Model already loaded.")

    def model_prediction(self, img):
        if not self.model:
            print("Model not loaded. Load or train the model first.")
            return
        
        prediction = self.model.predict(img)
        predicted_label = np.argmax(prediction)
        predicted_accuracy = round(np.max(prediction) * 100, 2)

        print('Predicted label: ', self.train_class[predicted_label])
        print(f'Percentage similarity of {predicted_accuracy}% towards {self.train_class[predicted_label]}')
        return self.train_class[predicted_label], predicted_accuracy

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0