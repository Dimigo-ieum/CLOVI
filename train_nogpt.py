import numpy as np
import tensorflow as tf


img_size = 224
batch_size = 32
val_split = 0.1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0)[..., None] # (N, 28, 28, 1)
y = np.concatenate([y_train, y_test], axis=0)

# upscale to RGB img_size x img_size:w
x = tf.image.resize(x, (img_size, img_size))
x = tf.image.grayscale_to_rgb(x)  # (N, H, W, 3)
x = tf.cast(x, tf.float32) / 255.0

N = x.shape[0]
print(N)


