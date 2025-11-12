import numpy as np
import tensorflow as tf


def preprocess(img, label):
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.image.grayscale_to_rgb(img)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

img_size = 224
batch_size = 32
val_split = 0.1
(data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.fashion_mnist.load_data()

# print("data train len: ", len(data_train), sep=" ")     # 40000
# print("labels train len: ", len(labels_train), sep=" ") # 40000
# print("asdf: ", data_train.shape) # (60000, 28, 28)
N = 60000
# print(labels_train) # array of integer 0-9

ds_train = tf.data.Dataset.from_tensor_slices((data_train, labels_train)).shuffle(10000, reshuffle_each_iteration=True)
ds_val  = tf.data.Dataset.from_tensor_slices((data_test, labels_test)).shuffle(10000, reshuffle_each_iteration=True)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_size, img_size, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds_train, validation_data=ds_val, epochs=5)


