import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tkinter import *
import tkinter as tk
# import win32gui
# import AppKit
# from PyObjCTools import AppHelper
# from PIL import ImageGrab, Image
# import tensorflow_datasets as tfds
from tensorflow import keras
from keras.datasets import mnist
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.python.keras import backend as 



physical_device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train sample')
print(x_test.shape[0],'test samples')

train_model = True

if train_model:
    batch_size = 128
    num_classes = 10
    epochs =3
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    hist = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
    print("The model has successfully trained")
    score = model.evaluate(x_test,y_test,verbose =0)
    print("'Test loss: ",score[0])
    print('Test accuracy: ',score[1])
    model.save('mnist.h5')
    print("Saving the model as mnist.h5")
   
else:
    model = load_model('mnist.h5')

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    img = 1 - img
    #predicting
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=400, height=400, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.btn_classify = tk.Button(self, text = "Recognise", command =self.classify_handwriting) 
        self.clear_button= tk.Button(self, text = "Clear",command = self.clear_all)
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.btn_classify.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        # HWND = self.canvas.winfo_id()
        # # rect = win32gui.GetWindowRect(HWND)
        # rect = AppKit.NSView.viewWithHandle(HWND)
        # im = ImageGrab.grab(rect)
        # digit, acc = predict_digit(im)
        # self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
        x = self.winfo_x() + self.canvas.winfo_x()
        y = self.winfo_y() + self.canvas.winfo_y()
        width = self.canvas.winfo_reqwidth()
        height = self.canvas.winfo_reqheight()
        im = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
app = App()
mainloop()
