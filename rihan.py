import datetime
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train / 255.0 , x_test /255.0

print(y_train[20])
plt.imshow(x_train[20])
plt.show()

models=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')])


models.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
log_dir="logs/"+datetime.datetime.now().strftime("%Ym%d-%H%M%S")
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath="model checkpoint.h5",save_best_only=True)
history=models.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test),callbacks=[tensorboard_callback,checkpoint_callback])
print("original models,accuracy:{:5.2f}%.format(100*acc)")