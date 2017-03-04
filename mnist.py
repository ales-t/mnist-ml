#!/usr/bin/env python3

import tensorflow as tf
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

class MNISTModel(object):
    """Train and evaluate deep learning models on the MNIST dataset."""

    def __init__(self, batch_size, batches, model_type, quiet):
        self._batch_size = batch_size
        self._batches = batches
        self._quiet = quiet

        if self._quiet:
            # optionally supress reporting from mnist.input_data
            # which writes to STDOUT
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        # the data is automatically rescaled to [0,1] float values (instead of 256 ints)
        self._data = input_data.read_data_sets('MNIST_data', one_hot=True)

        if self._quiet:
            # restore original stdout
            sys.stdout = old_stdout


        if model_type == "basic_cnn":
            self._vars = self._build_basic_model()
        elif model_type == "highway_cnn":
            self._vars = self._build_highway_model()
        else:
            raise Exception("Unknown model type: " + model_type)

    def _build_basic_model(self):
        """Multilayer CNN followed by a fully connected layer and softmax."""

        input_dim = 784 # images are 28x28
        output_dim = 10 # 10 possible labels

        x = tf.placeholder(tf.float32, shape=[None, input_dim]) # input
        y_ = tf.placeholder(tf.float32, shape=[None, output_dim]) # true label
    
        # make the instances two-dimensional for convolution
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    
        # first convolution layer
        h_conv1 = MNISTModel._conv_layer(x_image, 5, 1, 32)

        # max pooling, window size 2, stride 2
        h_pool1 = MNISTModel._max_pool(h_conv1, 2, 2)
    
        # second convolution layer
        h_conv2 = MNISTModel._conv_layer(h_pool1, 5, 32, 64)

        # max pooling, window size 2, stride 2
        h_pool2 = MNISTModel._max_pool(h_conv2, 2, 2)
    
        # fully connected layer (weight matrix, bias)
        W_fc1 = MNISTModel._weight_variable([7 * 7 * 64, 1024])
        b_fc1 = MNISTModel._bias_variable([1024])
    
        # flatten the output of convlayers for the fully connected layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        # RELU nonlinearity
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
        # drop-out from the fully connected layer
        use_dropout = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, 1 - use_dropout * 0.5)
    
        # final (readout) layer
        W_fc2 = MNISTModel._weight_variable([1024, output_dim])
        b_fc2 = MNISTModel._bias_variable([output_dim])

        # y contains logits of the possible labels
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # pass TF placeholder variables and output
        return { 'x': x, 'y': y, 'y_': y_, 'use_dropout': use_dropout }

    def _build_highway_model(self):
        """Highway network based on https://arxiv.org/abs/1507.06228. This
        implementation tries to follow the original code:

        https://github.com/flukeskywalker/highway-networks/blob/master/examples/highways/mnist-10layers/mnist_network.prototxt
        """

        input_dim = 784 # images are 28x28
        output_dim = 10 # 10 possible labels

        x = tf.placeholder(tf.float32, shape=[None, input_dim]) # input
        y_ = tf.placeholder(tf.float32, shape=[None, output_dim]) # true label
        use_dropout = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, [-1, 28, 28, 1]) # 2-dimensional for conv layers

        # drop-out input
        x_drop = tf.nn.dropout(x_image, 1 - use_dropout * 0.2)

        # first (standard) convolution layer, no padding
        h_conv1 = MNISTModel._conv_layer(x_drop, 5, 1, 16, padding='VALID')
    
        # highway CNN layers, no max pooling
        h_conv2 = MNISTModel._highway_conv_layer(h_conv1, 24, 3, 16)
        h_conv3 = MNISTModel._highway_conv_layer(h_conv2, 24, 3, 16)

        # max pool
        h_pool1 = MNISTModel._max_pool(h_conv3, size=3, stride=2)
        pool1_drop = tf.nn.dropout(h_pool1, 1 - use_dropout * 0.3)

        # second set of highway CNN layers
        h_conv4 = MNISTModel._highway_conv_layer(pool1_drop, 12, 3, 16)
        h_conv5 = MNISTModel._highway_conv_layer(h_conv4, 12, 3, 16)
        h_conv6 = MNISTModel._highway_conv_layer(h_conv5, 12, 3, 16)

        # max pool
        h_pool2 = MNISTModel._max_pool(h_conv6, size=3, stride=2)
        pool2_drop = tf.nn.dropout(h_pool2, 1 - use_dropout * 0.4)

        # final set of highway CNN layers
        h_conv7 = MNISTModel._highway_conv_layer(pool2_drop, 6, 3, 16)
        h_conv8 = MNISTModel._highway_conv_layer(h_conv7, 6, 3, 16)
        h_conv9 = MNISTModel._highway_conv_layer(h_conv8, 6, 3, 16)

        # final max pool
        h_pool3 = MNISTModel._max_pool(h_conv9, size=2, stride=2)
        pool3_drop = tf.nn.dropout(h_pool3, 1 - use_dropout * 0.5)
        pool3_drop_flat = tf.reshape(pool3_drop, [-1, 3 * 3 * 16])

        # final (readout) layer
        W_fc2 = MNISTModel._weight_variable([3 * 3 * 16, output_dim])
        b_fc2 = MNISTModel._bias_variable([output_dim])
    
        y = tf.matmul(pool3_drop_flat, W_fc2) + b_fc2

        return { 'x': x, 'y': y, 'y_': y_, 'use_dropout': use_dropout }

    def train(self, session):
        """Train the model for the specified number of batches."""
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=self._vars['y_'], logits=self._vars['y'])
        loss = tf.reduce_mean(xent)

        train_step = tf.train.AdamOptimizer().minimize(loss)
        session.run(tf.global_variables_initializer()) 

        for i in range(self._batches):

            if not self._quiet:
                # by default, give the user some sense of progress
                sys.stderr.write('.')
                sys.stderr.flush()

            batch = self._data.train.next_batch(self._batch_size)
            session.run(
                [train_step],
                feed_dict={ 
                    self._vars['x']: batch[0], 
                    self._vars['y_']: batch[1], 
                    self._vars['use_dropout']: 1.0, # float instead of boolean for simplicity
                })

    def _test_model(self, session, dataset):
        """Test the current model against the specified data set."""
        is_correct = tf.equal(tf.argmax(self._vars['y'], 1), tf.argmax(self._vars['y_'],1))
        accuracy_fn = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        accuracy = session.run(
            [accuracy_fn], 
            feed_dict={
                self._vars['x']: dataset.images, 
                self._vars['y_']: dataset.labels, 
                self._vars['use_dropout']: 0.0, # no drop-out when testing
            })

        return accuracy

    def validate(self, session):
        """Test the model on the validation data."""
        return self._test_model(session, self._data.validation)

    def eval(self, session):
        """Test the model on the official test set."""
        return self._test_model(session, self._data.test)

    @staticmethod
    def _weight_variable(shape, mean=0.0, stddev=0.1):
        """Shorthand for initializing a weight matrix/vector. 
        Uses random sampling from a truncated Gaussian distribution."""
        initial = tf.truncated_normal(shape, mean, stddev)
        return tf.Variable(initial)
    
    @staticmethod
    def _bias_variable(shape, init=0.1):
        """Shorthand for bias vector initialization (Which is recommended
        to be constant)."""
        initial = tf.constant(init, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv_layer(x, filter_size, in_filters, out_filters, nonlinearity=tf.nn.relu, padding='SAME'):
        """Shorthand for defining a convolutional layer."""
        W = MNISTModel._weight_variable([filter_size, filter_size, in_filters, out_filters])
        b = MNISTModel._bias_variable([out_filters])
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    
        return nonlinearity(conv + b)
    
    @staticmethod
    def _max_pool(x, size, stride):
        """Shorthand for defining a max pooling operation."""
        return tf.nn.max_pool(x, 
                              ksize=[1, size, size, 1], 
                              strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def _highway_conv_layer(x, size, filter_size, filter_count, gate_bias=-1.0):
        """A convolutional highway layer. Creates a standard conv layer, flattens
        its output and sends it to a _highway_layer()."""

        # create the conv layer with specified parameters
        conv = MNISTModel._conv_layer(x, filter_size, filter_count, filter_count)

        # flatten the tensor (originally height x width x filter_count)
        conv_flat = tf.reshape(conv, [-1, size * size * filter_count])

        # flatten the original input (originally height x width x filter_count)
        x_flat = tf.reshape(x, [-1, size * size * filter_count])

        # create the highway layer
        highway = MNISTModel._highway_layer(x_flat, size * size * filter_count, conv_flat, gate_bias)

        # reshape the flat output back into height x width x filter_count
        return tf.reshape(highway, [-1, size, size, filter_count])

    @staticmethod
    def _highway_layer(x, size, activation, gate_bias=-1.0):
        """Highway network layer as defined by Srivastava et al. 2015. 
        Activation should contain the transformed input x, including 
        the non-linearity."""
        W_t = MNISTModel._weight_variable([size, size])
        # use negative bias for transform gates
        b_t = MNISTModel._bias_variable([size], init=gate_bias)
    
        # gate values
        g = tf.nn.sigmoid(tf.matmul(x, W_t) + b_t)
    
        return tf.multiply(g, activation) + tf.multiply(1 - g, x)
