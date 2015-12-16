import tensorflow as tf
import numpy as np
import cv2

## Functions for the neural networks
# Weight and bias (random) initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

### Functions to transform image data into mnist format
# Converts image to mnist data format for digits
def get_mnist_format(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = np.float32(np.array([img.flatten()]))
    img /= np.amax(img)
    return img
