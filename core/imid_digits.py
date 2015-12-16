# Loads the trained model and idenfies digits from an image

import tensorflow as tf
import numpy as np
import input_data
import cv2
from functions import *

class imid_digits:
    # Constructor
    def __init__(self):
        # Interactive session
        self.sess = tf.InteractiveSession()

        # Nodes for the input images and target output classes
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", shape=[None, 10])

        # Variables to calculate hypothesis and min function
        self.W = tf.Variable(tf.zeros([784, 10])) # Weigh values
        self.b = tf.Variable(tf.zeros([10])) # Out bias values (to prevent overfitting)

        # Start our session
        self.sess.run(tf.initialize_all_variables())

        # Neural Network
        # First convolution layer
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        # Second convlution layer
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 =max_pool_2x2(self.h_conv2)

        # Densely connected layer
        self.W_fc1 = weight_variable([7*7*64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)


        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # Softmax layer
        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        # Correct outputs
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        # Minimize function
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_,1))

        # Evaluation
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.sess.run(tf.initialize_all_variables())

    # Trains and saves model
    def train_and_save_model(self):
        # Our training data
        mnist = input_data.read_data_sets('../data/MNIST_digits', one_hot=True)

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0
                })
                print("step %d, training accuracy %g"%(i, train_accuracy))
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

        # Saves path
        save_path = saver.save(sess, "../trained_models/mnist_digits.ckpt")
        print("Model saved in file: ", save_path)

    # Loads saved model
    def load_model(self):
        # Loads saved model
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "../trained_models/mnist_digits.ckpt")

    def identify(self, img):
        img = get_mnist_format(img)

        pred = tf.argmax(self.y_conv, 1)
        print(pred.eval(feed_dict={self.x:img, self.keep_prob: 0.5}))
