#!/usr/bin/env python3

import tensorflow as tf
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

class MNISTModel(object):

    def __init__(self, batch_size, batches, dropout_prob):
        self._batch_size = batch_size
        self._batches = batches
        self._keep_prob = 1.0 - dropout_prob

        self._data = input_data.read_data_sets('MNIST_data', one_hot=True)
        self._vars = self._build_model()

    def _build_model(self):
        input_dim = 784
        output_dim = 10

        x = tf.placeholder(tf.float32, shape=[None, input_dim])
        y_ = tf.placeholder(tf.float32, shape=[None, output_dim])
    
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    
        # first convolution layer
        W_conv1 = MNISTModel._weight_variable([5, 5, 1, 32])
        b_conv1 = MNISTModel._bias_variable([32])
    
        h_conv1 = tf.nn.relu(MNISTModel._conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = MNISTModel._max_pool_2x2(h_conv1)
    
        # second convolution layer
        W_conv2 = MNISTModel._weight_variable([5, 5, 32, 64])
        b_conv2 = MNISTModel._bias_variable([64])
    
        h_conv2 = tf.nn.relu(MNISTModel._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = MNISTModel._max_pool_2x2(h_conv2)
    
        # fully connected layer
        W_fc1 = MNISTModel._weight_variable([7 * 7 * 64, 1024])
        b_fc1 = MNISTModel._bias_variable([1024])
    
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
        # drop-out from the fully connected layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
        # final (readout) layer
        W_fc2 = MNISTModel._weight_variable([1024, output_dim])
        b_fc2 = MNISTModel._bias_variable([output_dim])
    
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return { 'x': x, 'y': y, 'y_': y_, 'keep_prob': keep_prob }

    def train(self, session):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=self._vars['y_'], logits=self._vars['y'])
        loss = tf.reduce_mean(xent)

        train_step = tf.train.AdamOptimizer().minimize(loss)
        session.run(tf.global_variables_initializer()) 

        for i in range(self._batches):
            sys.stderr.write('.')
            sys.stderr.flush()
            batch = self._data.train.next_batch(self._batch_size)
            session.run(
                [train_step],
                feed_dict={ 
                    self._vars['x']: batch[0], 
                    self._vars['y_']: batch[1], 
                    self._vars['keep_prob']: 0.5,
                })

    def _test_model(self, session, dataset):
        is_correct = tf.equal(tf.argmax(self._vars['y'], 1), tf.argmax(self._vars['y_'],1))
        accuracy_fn = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        accuracy = session.run(
            [accuracy_fn], 
            feed_dict={
                self._vars['x']: dataset.images, 
                self._vars['y_']: dataset.labels, 
                self._vars['keep_prob']: 1, # no drop-out when testing
            })

        return accuracy

    def validate(self, session):
        return self._test_model(session, self._data.validation)

    def eval(self, session):
        return self._test_model(session, self._data.test)

    @staticmethod
    def _weight_variable(shape, mean=0.0, stddev=0.1):
        initial = tf.truncated_normal(shape, mean, stddev)
        return tf.Variable(initial)
    
    @staticmethod
    def _bias_variable(shape, init=0.1):
        initial = tf.constant(init, shape=shape)
        return tf.Variable(initial)
    
    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def highway_layer(x, size, activation, gate_bias=-1.0):
        W = MNISTModel._weight_variable([size, size])
        # use negative bias for transform gates
        b = MNISTModel._bias_variable([size], init=gate_bias)
    
        g = tf.nn.sigmoid(tf.matmul(x, W_t) + b_t)
    
        nonlinear = activation(x)
    
        return tf.multiply(g, nonlinear) + tf.multiply(1 - g, x)
