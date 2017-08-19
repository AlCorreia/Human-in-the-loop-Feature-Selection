import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.ops.losses.losses_impl import Reduction
from base_model import Model


class Bernoulli_Net(Model):
    """ Creates a deep learning model. """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, frame_size, learning_rate, directory, meta=None):
        # Initialize the Model object
        super().__init__(directory, learning_rate, num_filters, num_features, frame_size)

        if meta is not None:
            # Load pre-trained model
            self.saver = tf.train.import_meta_graph(directory + '/' + meta)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))
            graph = tf.get_default_graph()
            self.X = graph.get_tensor_by_name("X:0")
            self.label = graph.get_tensor_by_name("y:0")
            self.probs = graph.get_tensor_by_name("attention/prediction:0")
            self.predictions = graph.get_tensor_by_name("dnn/prediction:0")
        else:
            input_size = self.X.get_shape()[1]
            batch_size = self.X.get_shape()[0]

            # Build the graph for the neural classifier
            conv = tf.layers.conv2d(
                          inputs= tf.reshape(self.X, [-1, 60, 60, 1]),
                          filters=self.num_filters,
                          kernel_size=[5, 5],
                          padding="same",
                          strides=(3, 3),
                          name='conv1',
                          activation=tf.nn.relu)
                        #   kernel_regularizer=l2_regularizer,
                        #   bias_regularizer=l2_regularizer)

            # conv = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1],
            #             strides=[1, 3, 3, 1], padding='SAME')
            conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv1")
            features = tf.reshape(conv, [-1, self.num_features, self.num_filters])
            flat_features = tf.reshape(conv, [-1, self.num_features*self.num_filters])
            # Build the controller to estimate the probability of each feature
            probs_logits, self.probs, policy_reg_loss = self.build_net(
                input=flat_features,
                layer_sizes=[self.num_features*self.num_filters, 128, self.num_features],
                nonlinearity=tf.nn.relu,
                last_activation=tf.nn.sigmoid,
                scope='attention',
                phase=self.phase)

            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "attention")

            # Define a Bernoulli distribution based on the probabilities estimated by the controller
            self.bernoulli = tf.contrib.distributions.Bernoulli(logits=probs_logits, allow_nan_stats=False, dtype=tf.float32)
            self.attention = tf.placeholder(tf.float32, shape=[None, self.num_features], name="a")
            # Calculate the probability of each attention vector
            sample_probs = self.bernoulli.prob(self.attention)

            # Define summaries to monitor the agent
            attention_summary = tf.summary.histogram('a', self.attention)
            probability_summary = tf.summary.histogram('probability', self.probs)
            sample_probability_summary = tf.summary.histogram('sample_probability', sample_probs)

            # Calculate regularizers based on the sparsity of the distribution of probabilities
            batch_mean, batch_variance = tf.nn.moments(self.probs, axes=[0])
            example_mean, example_variance = tf.nn.moments(self.probs, axes=[1])
            Lb = tf.reduce_mean(tf.square(batch_mean - 0.20))
            Le = tf.reduce_mean(tf.square(example_mean - 0.20))
            Lve = tf.reduce_mean(example_variance)
            Lvb = tf.reduce_mean(batch_variance)

            Lb_summary = tf.summary.scalar('Lb', Lb)
            Le_summary = tf.summary.scalar('Le', Le)
            Lv_summary = tf.summary.scalar('Lve', Lve)
            Lv_summary = tf.summary.scalar('Lvb', Lvb)

            # Multiply the features by the gates defined by attention
            zoom_X = tf.multiply(features, tf.reshape(self.attention, [-1, self.num_features, 1]), name='Zoom_X')
            drop = tf.nn.dropout(zoom_X, keep_prob=self.keep_prob)
            flat_zoom_X = tf.reshape(zoom_X, [-1, self.num_features*self.num_filters])

            self.logits, self.predictions, class_reg_loss = self.build_net(
                input=flat_zoom_X,
                layer_sizes=[self.num_features*self.num_filters] + layer_sizes + [self.num_classes],
                nonlinearity=tf.nn.relu,
                scope='dnn',
                phase=self.phase)

            classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dnn")

            # Define the loss function
            cost = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.label,
                    name='loss_function')

            # feedback_cost = tf.losses.log_loss(
            #         labels=self.feedback,
            #         predictions=self.probs)

            past_cost = tf.Variable(tf.zeros([]), name='past_cost')

            cost_mean = tf.reduce_mean(cost)
            cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))

            log_probability = tf.log(sample_probs + 10e-20)
            log_probability = tf.verify_tensor_all_finite(log_probability, "Undefined log probability")
            log_probability = tf.reduce_sum(log_probability, axis=1)

            # Define the loss function for the classifier
            self.policy_loss = tf.reduce_mean(tf.multiply(tf.stop_gradient(cost-past_cost), log_probability)) + (Le + Lb) - (Lve + Lvb)
            self.loss = cost_mean

            self.pre_train_step = tf.contrib.layers.optimize_loss(
                    cost_mean, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], name='PRE_TRAIN')

            # Define the train step
            self.train_step = [
                tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], name='CLASSIFIER'),

                tf.contrib.layers.optimize_loss(
                    self.policy_loss, global_step=self.global_step, increment_global_step=False, learning_rate=0.05*self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=policy_vars, name='REINFORCE')
            ]

            feedback_cost = tf.losses.cosine_distance(
                    labels=self.feedback,
                    predictions=tf.reshape(self.attention, [-1, self.num_features]),
                    dim=1,
                    reduction=Reduction.NONE)
            feedback_cost_total = tf.reduce_mean(feedback_cost) + self.policy_loss

            self.feedback_train_step = [
                tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], name='CLASSIFIER2'),

                tf.contrib.layers.optimize_loss(
                        feedback_cost_total, global_step=self.global_step, increment_global_step=False, learning_rate=self.learning_rate, optimizer=optimizer,
                        summaries=["gradients"], variables=[policy_vars], name='FEEDBACK_TRAIN')
            ]

            tf.assign(past_cost, tf.reduce_mean(cost))

            self.add_logging()  # Monitor loss, accuracy and auc
            # Initialize all the local variables (used to calculate auc & accuracy)
            local_init = tf.local_variables_initializer()
            self.sess.run(local_init)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            input_images = tf.summary.image("Visualize_Input", tf.reshape(self.X, [10, 60, 60, 1]), max_outputs=10)
            prob_images = tf.summary.image("Probability_Distribution", tf.reshape(self.probs, [10, 20, 20, 1]), max_outputs=10)
            feedback_images = tf.summary.image("Feedback_Images", tf.reshape(self.feedback, [10, 20, 20, 1]), max_outputs=10)
            self.merged_images = tf.summary.merge([input_images, prob_images, feedback_images])


class Cat_Net(Model):
    """ Creates a deep learning model. """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, frame_size, num_cat, learning_rate, directory, meta=None):
        # Initialize the Model object
        super().__init__(directory, learning_rate, num_filters, num_features, frame_size)

        if meta is not None:
            # Load pre-trained model
            self.saver = tf.train.import_meta_graph(directory + '/' + meta)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))
            graph = tf.get_default_graph()
            self.X = graph.get_tensor_by_name("X:0")
            self.label = graph.get_tensor_by_name("y:0")
            self.probs = graph.get_tensor_by_name("attention/prediction:0")
            self.predictions = graph.get_tensor_by_name("dnn/prediction:0")
        else:
            input_size = self.X.get_shape()[1]
            batch_size = self.X.get_shape()[0]

            conv = tf.layers.conv2d(
                inputs= tf.reshape(self.X, [-1, 60, 60, 1]),
                filters=self.num_filters,
                kernel_size=[5, 5],
                padding="same",
                strides=(3, 3),
                name='conv1',
                activation=tf.nn.relu)

            conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv1")
            self.features = tf.reshape(conv, [-1, self.num_features, self.num_filters])
            flat_features = tf.reshape(conv, [-1, self.num_features*self.num_filters])
            # Build the controller to estimate the probability of each feature
            probs_logits, _, policy_reg_loss = self.build_net(
                input=flat_features,
                layer_sizes=[self.num_features*self.num_filters, 128, self.num_features*num_cat],
                nonlinearity=tf.nn.relu,
                last_activation=tf.nn.sigmoid,
                scope='attention',
                phase=self.phase)

            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "attention")

            self.probs = tf.nn.softmax(tf.reshape(probs_logits, [-1, self.num_features, num_cat]))
            self.shape = tf.shape(self.probs)

            # Define a categorical distribution based on the probabilities estimated by the controller
            self.categorical = tf.contrib.distributions.Categorical(probs=tf.reshape(self.probs, [-1, self.num_features, num_cat]), allow_nan_stats=False, dtype=tf.float32)
            self.attention = tf.placeholder(tf.float32, shape=[None, self.num_features], name="a")

            tf.assert_rank(self.attention, 2)
            attention_summary = tf.summary.histogram('attention', self.attention)
            # Calculate the probability of each attention vector
            sample_probs = self.categorical.prob(tf.to_int32(self.attention))
            probability_summary = tf.summary.histogram('probability', self.probs)
            sample_probability_summary = tf.summary.histogram('sample_probability', sample_probs)

            weights = tf.linspace(0.0, 1.0, num_cat, name="weights")
            weights = tf.reshape(weights, [-1, num_cat])

            self.categorical_sample = tf.reshape(self.categorical.sample()/num_cat, [-1, self.num_features, 1])
            # Multiply the features by the gates defined by attention
            weighthed_probs = tf.reshape(tf.matmul(weights, tf.reshape(self.probs, [num_cat, -1])), [-1, self.num_features, 1])
            attention_summary = tf.summary.histogram('attention', self.attention)

            # Calculate regularizers based on the sparsity of the distribution of probabilities
            batch_mean, batch_variance = tf.nn.moments(weighthed_probs, axes=[0])
            example_mean, example_variance = tf.nn.moments(weighthed_probs, axes=[1])
            Lb = tf.reduce_mean(tf.square(batch_mean - 0.20))
            Le = tf.reduce_mean(tf.square(example_mean - 0.20))
            Lve = tf.reduce_mean(example_variance)
            Lvb = tf.reduce_mean(batch_variance)

            Lb_summary = tf.summary.scalar('Lb', Lb)
            Le_summary = tf.summary.scalar('Le', Le)
            Lv_summary = tf.summary.scalar('Lve', Lve)
            Lv_summary = tf.summary.scalar('Lvb', Lvb)

            # Multiply the features by the gates defined by attention
            zoom_X = tf.multiply(self.features, tf.reshape(self.attention, [-1, self.num_features, 1]), name='Zoom_X')
            flat_zoom_X = tf.reshape(zoom_X, [-1, self.num_features*self.num_filters])

            # Build the graph for the neural classifier
            self.logits, self.predictions, class_reg_loss = self.build_net(
                input=flat_zoom_X,
                layer_sizes=[self.num_features*self.num_filters] + layer_sizes + [self.num_classes],
                nonlinearity=tf.nn.relu,
                scope='dnn',
                phase=self.phase)

            classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dnn")

            # Define the loss function
            cost = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.label,
                    name='loss_function')

            past_cost = tf.Variable(tf.zeros([]), name='past_cost')

            cost_mean = tf.reduce_mean(cost)
            cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))

            log_probability = tf.log(sample_probs + 10e-20)
            log_probability = tf.verify_tensor_all_finite(log_probability, "Undefined log probability")
            log_probability = tf.reduce_sum(log_probability, axis=1)

            self.loss = cost_mean

            self.policy_loss = tf.reduce_mean(tf.multiply(tf.stop_gradient(cost-past_cost), log_probability)) + (Le + Lb) - (Lve + Lvb)
            # Passing global_step to minimize() will increment it at each step.
            # Define the train step
            self.pre_train_step = tf.contrib.layers.optimize_loss(
                    cost_mean, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], name='PRE_TRAIN')

            self.train_step = [
                tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer='Adagrad',
                    summaries=["gradients"], variables=classifier_vars, name='CLASSIFIER'),

                tf.contrib.layers.optimize_loss(
                    self.policy_loss, global_step=self.global_step, increment_global_step=False, learning_rate=0.01*self.learning_rate, optimizer='Adagrad',
                    summaries=["gradients"], variables=policy_vars, name='REINFORCE')
            ]


            feedback_cost = tf.losses.cosine_distance(
                    labels=self.feedback,
                    predictions=tf.reshape(self.attention, [-1, self.num_features]),
                    dim=1,
                    reduction=Reduction.NONE)

            feedback_cost_total = tf.reduce_mean(feedback_cost) + self.policy_loss

            self.feedback_train_step = [
                tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], name='CLASSIFIER2'),

                tf.contrib.layers.optimize_loss(
                        feedback_cost_total, global_step=self.global_step, increment_global_step=False, learning_rate=self.learning_rate, optimizer=optimizer,
                        summaries=["gradients"], variables=[policy_vars], name='FEEDBACK_TRAIN')
            ]

            tf.assign(past_cost, tf.reduce_mean(cost))

            self.add_logging()  # Monitor loss, accuracy and auc
            # Initialize all the local variables (used to calculate auc & accuracy)
            local_init = tf.local_variables_initializer()
            self.sess.run(local_init)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            input_images = tf.summary.image("Visualize_Input", tf.reshape(self.X, [10, 60, 60, 1]), max_outputs=10)
            prob_images = tf.summary.image("Probability_Distribution", tf.reshape(weighthed_probs, [10, 20, 20, 1]), max_outputs=10)
            feedback_images = tf.summary.image("Feedback_Images", tf.reshape(self.feedback, [10, 20, 20, 1]), max_outputs=10)

            self.merged_images = tf.summary.merge([input_images, prob_images, feedback_images])
