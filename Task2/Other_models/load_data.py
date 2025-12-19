import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Save images to disk
os.makedirs('mnist_images/train', exist_ok=True)
os.makedirs('mnist_images/test', exist_ok=True)

# Save training images
for i in range(len(x_train)):
    img = Image.fromarray(x_train[i])
    img.save(f'mnist_images/train/{y_train[i]}_{i}.png')

# Save test images
for i in range(len(x_test)):
    img = Image.fromarray(x_test[i])
    img.save(f'mnist_images/test/{y_test[i]}_{i}.png')