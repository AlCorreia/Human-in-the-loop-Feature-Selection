import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.framework import graph_util
from sklearn.metrics import cohen_kappa_score


class Model(object):
    """ General model class. """
    def __init__(self, directory, learning_rate, num_filters, num_features, frame_size):
        """ Initializes the model and creates a neww seesion.

            Parameters
            ----------
            directory : str
                The folder where the model is stored.
            learning_rate : float
                Defines the learning rate for gradient descent.

        """
        self.directory = directory
        # Create a tensorflow session for the model
        self.sess = tf.Session()
        # Define the learning rate for the neural net
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=1000,
                                                        decay_rate=0.96,
                                                        staircase=True)
        # self.learning_rate = learning_rate
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout')

        self.num_filters = num_filters
        self.num_features = num_features
        self.num_classes = 10
        self.frame_size = frame_size
        # Define placeholder for the inputs X, the labels y and the users feedback
        self.label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
        self.X = tf.placeholder(tf.float32, shape=[None, self.frame_size**2], name="X")
        self.feedback = tf.placeholder(tf.float32, shape=[None, self.num_features], name="feedback")
        # Create a placeholder that defines the phase: training or testing
        self.phase = tf.placeholder(tf.bool, name='phase')

    def explain(self, X):
        feed_dict = {}
        feed_dict['phase:0'] = 0
        feed_dict['X:0'] = X
        # Run the evaluation
        probs = self.sess.run(tf.reduce_mean(self.probs, axis=0),
                              feed_dict=feed_dict)
        return probs

    def pre_train(self, X, y, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # The attention tensor is set to ones during pre-training
        feed_dict['a:0'] = np.ones([X.shape[0], self.num_features])
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        feed_dict['phase:0'] = 1
        # Run the training step and write the results to Tensorboard
        summary, _ = self.sess.run([self.merged, self.pre_train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        summary = self.sess.run(self.merged_images,
                                feed_dict={self.X: X[0:10], self.feedback: np.ones([10, self.num_features]), self.attention: np.ones([10, self.num_features])})
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')

    def train(self, X, y, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(tf.reshape(self.distribution.sample(), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        feed_dict['phase:0'] = 1
        # Run the training step and write the results to Tensorboard
        summary, _ = self.sess.run([self.merged, self.train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        summary = self.sess.run(self.merged_images,
                                feed_dict={self.X: X[0:10],
                                           self.feedback: np.ones([10, self.num_features]), self.attention: attention[0:10]})
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')

    def feedback_train(self, X, y, feedback, dropout=0.5):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            feedback: numpy array [batch_size, num_features]
            dropout: float in [0, 1]
                The probability of keeping the neurons active

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['dropout:0'] = dropout
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(tf.reshape(self.distribution.sample(), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = feedback
        feed_dict['phase:0'] = 1
        # Run the training step and write the results to Tensorboard
        summary, _ = self.sess.run([self.merged, self.feedback_train_step],
                                   feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        summary = self.sess.run(self.merged_images,
                                feed_dict={self.X: X[0:10], self.feedback: feedback[0:10], self.attention: attention[0:10]})
        self.train_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')

    def evaluate(self, X, y, pre_trainining=False):
        """ Evaluates X and compares to the correct label y.

            Parameters
            ----------
            X : numpy array [batch_size, num_raw_features]
            y : numpy array [batch_size, num_classes]
            pre_trainining: boolean

        """
        # Combine the input dictionaries for all the features models
        feed_dict = {}
        feed_dict['X:0'] = X
        feed_dict['y:0'] = y
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        if pre_trainining:
            attention = np.ones([X.shape[0], self.num_features])
        else:
            attention = self.sess.run(tf.reshape(self.distribution.sample(), [-1, self.num_features]), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['phase:0'] = 0
        # Run the evaluation and write the results to Tensorboard
        summary = self.sess.run(self.merged,
                                feed_dict=feed_dict)
        self.test_writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Run image summary and write the results to Tensorboard
        summary = self.sess.run(self.merged_images,
                                feed_dict={self.X: X[0:10], self.feedback: np.ones([10, self.num_features]), self.attention: attention[0:10]})
        self.test_writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def predict(self, X):
        feed_dict = {}
        feed_dict["phase:0"] = 0
        feed_dict["X:0"] = X
        # Sample the attention vector from the Bernoulli distribution
        attention = self.sess.run(self.distribution.sample(), feed_dict=feed_dict)
        feed_dict['a:0'] = attention
        feed_dict['feedback:0'] = np.ones([X.shape[0], self.num_features])
        # Run the evaluation
        return self.sess.run(self.predictions, feed_dict=feed_dict)

    def save(self):
        self.saver.save(self.sess, self.directory + '/model.ckpt', global_step=self.sess.run(self.global_step))

    def confusion_matrix(self, X, y):
        predictions = self.predict(X)
        predictions = tf.argmax(predictions, axis=1)
        return self.sess.run(tf.confusion_matrix(labels=np.argmax(y, axis=1), predictions=predictions))

    def kappa(self, X, y):
        predictions = self.predict(X)
        predictions = np.argmax(predictions, axis=1)
        y = np.argmax(y, axis=1)
        return cohen_kappa_score(y, predictions)

    def add_logging(self, config=None):
        """ Creates summaries for accuracy, auc and loss. """

        # Calculate the accuracy for multiclass classification
        correct_prediction = tf.equal(tf.argmax(self.label, axis=1), tf.argmax(self.predictions, axis=1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        # Calculate the confusion matrix
        confusion = tf.confusion_matrix(labels=tf.argmax(self.label, axis=1), predictions=tf.argmax(self.predictions, axis=1))
        confusion = tf.cast(confusion, dtype=tf.float32)
        # Calculate the kappa accuracy
        sum0 = tf.reduce_sum(confusion, axis=0)
        sum1 = tf.reduce_sum(confusion, axis=1)
        expected_accuracy = tf.reduce_sum(tf.multiply(sum0, sum1)) / (tf.square(tf.reduce_sum(sum0)))
        expected_accuracy_summary = tf.summary.scalar('expected_accuracy', expected_accuracy)
        w_mat = np.zeros([self.num_classes, self.num_classes], dtype=np.int)
        w_mat += np.arange(self.num_classes)
        w_mat = np.abs(w_mat - w_mat.T)
        k = (self.accuracy - expected_accuracy) / (1 - expected_accuracy)
        # Calculate the precision
        _, precision = tf.metrics.precision(
                                    labels=self.label,
                                    predictions=self.predictions)
        # Add the summaries
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        kappa_accuracy = tf.summary.scalar('kappa', k)
        precision_summary = tf.summary.scalar('precision', precision)
        loss_summary = tf.summary.scalar('loss', self.loss)
        learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)

        # Combine all the summaries
        self.merged = tf.summary.merge_all()
        # Create summary writers for train and test set
        self.train_writer = tf.summary.FileWriter(self.directory + '/train',
                                                  self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.directory + '/test')

    def build_net(self, input, layer_sizes, scope, phase, nonlinearity=tf.nn.relu, last_activation=tf.nn.softmax, logging=True, batch_norm=False, reuse=False):
        """ Builds a dense DNN with the architecture defined by layer_sizes.

            Parameters
            ----------
            input: tf tensor
                The input to the neural net
            layer_sizes: list
                 List containing the size of all layers, including the input and output sizes.
                 [input_size, hidden_layers, output_size]
            scope: str
                The name of the variable scope, which refers to the type of model
            phase: tf placeholder (boolean)
                Holds the state of the model: training or testing
            nonlinearity: tf callable
                The nonlinearity applied between layers in the neural net
            last_activation: tf callable
                The nonlinearity that defines the output of the neural net
            logging: boolean
                Whether to log the weights in Tensorboard
            batch_norm: boolean
                Whether to apply batch normalization between layers
            reuse: boolean
                Whether the weights in neural net can be reused later on

            Returns
            ----------
            pre_activation: tf tensor
                The logits which are equivalent to the output of the DNN before the non-linearity
            predictions: tf tensor
                The output of the DNN
            reg_loss: tf tensor
                The regularization loss of the neural net

        """
        activation = input
        reg_loss = []
        # Define the DNN layers
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            for i, layer_size in enumerate(layer_sizes[1::], 1):
                if i < len(layer_sizes) - 1:
                    W = tf.get_variable(
                            name='W{0}'.format(i),
                            shape=[layer_sizes[i-1], layer_size],
                            initializer=tf.contrib.layers.xavier_initializer())
                    reg_loss.append(tf.nn.l2_loss(W))
                    b = tf.Variable(
                            tf.zeros([layer_size]), name='b{0}'.format(i))
                    if logging:
                        summ = tf.summary.histogram('W{0}'.format(i), W)
                        summm = tf.summary.scalar('W{0}_sparsity'.format(i), tf.nn.zero_fraction(W))
                else:
                    with tf.variable_scope("last"):
                        W = tf.get_variable(
                                name='W{0}'.format(i),
                                shape=[layer_sizes[i-1], layer_size],
                                initializer=tf.contrib.layers.xavier_initializer())
                        reg_loss.append(tf.nn.l2_loss(W))
                        b = tf.Variable(
                                tf.zeros([layer_size]), name='b{0}'.format(i))
                        if logging:
                            summ = tf.summary.histogram('W{0}'.format(i), W)
                            summm = tf.summary.scalar('W{0}_sparsity'.format(i), tf.nn.zero_fraction(W))

                if i < len(layer_sizes) - 1:
                    pre_activation = tf.add(tf.matmul(activation, W), b,
                                            name='pre_activation{0}'.format(i))
                    activation = nonlinearity(pre_activation,
                                              name='activation{0}'.format(i))

                    activation = tf.nn.dropout(activation, keep_prob=self.keep_prob)
                    if batch_norm:
                        activation = tf.contrib.layers.batch_norm(activation,
                                                                  center=True,
                                                                  scale=True,
                                                                  is_training=phase)
                else:
                    pre_activation = tf.add(tf.matmul(activation, W), b,
                                            name='logits')
                    # The last non-linearity is a softmax function
                    if last_activation is not None:
                        activation = last_activation(pre_activation, name='prediction')
                    else:
                        activation = tf.identify(pre_activation, name='prediction')
        return pre_activation, activation, reg_loss
