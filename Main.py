import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import datetime

# Load the dataset and preprocess the images
df = pd.read_csv("mnist_dataset.csv")
X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
X = X.astype("float32") / 255.0
y = to_categorical(df.iloc[:, 0])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile and train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Save model for later
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[tensorboard_callback, cp_callback])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

model.summary()

# Save the entire model as a SavedModel.
model.save('saved_model')
tfjs.converters.save_keras_model(model, 'models_tfjs')

#Tensorboard: tensorboard --logdir logs/fit

