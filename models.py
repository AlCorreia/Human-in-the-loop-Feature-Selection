import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.ops.losses.losses_impl import Reduction
from base_model import Model
from utils import *


class Bernoulli_Net(Model):
    """ Creates a deep learning model. """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, num_samples, frame_size, learning_rate, feedback_distance, directory, second_conv=False, meta=None):
        # Initialize the Model object
        super().__init__(directory, learning_rate, num_filters, num_features, frame_size, num_samples)

        if meta is not None:
            # Load pre-trained model
            self.saver = tf.train.import_meta_graph(directory + '/' + meta)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))
            graph = tf.get_default_graph()
            conv = graph.get_tensor_by_name("conv1:0")
        else:
            # Build the graph for the neural classifier
            conv = tf.layers.conv2d(
                          inputs= tf.reshape(self.X, [-1, frame_size, frame_size, 1]),
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
        self.distribution = tf.contrib.distributions.Bernoulli(logits=probs_logits, allow_nan_stats=False, dtype=tf.float32)
        dynamic_num_samples = tf.floordiv(tf.shape(self.attention)[0], tf.shape(self.X)[0])
        # Calculate the probability of each attention vector
        sample_probs = tf.reshape(self.distribution.prob(tf.reshape(self.attention, [dynamic_num_samples, -1, self.num_features])), [-1, self.num_features])

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
        features = tf.reshape(tf.tile(tf.reshape(features, [-1]), [dynamic_num_samples]), [-1, self.num_features, self.num_filters])
        self.attention = tf.cond(tf.greater(self.phase, 0), lambda: self.attention, lambda: tf.round(self.attention))
        zoom_X = tf.multiply(features, tf.reshape(self.attention, [-1, self.num_features, 1]), name='Zoom_X')

        if second_conv:
            # Multiply the features by the gates defined by attention
            zoom_X = tf.layers.conv2d(
                        inputs= tf.reshape(zoom_X, [-1, 20, 20, self.num_filters]),
                        filters=self.num_filters*2,
                        kernel_size=[5, 5],
                        padding="same",
                        strides=(1, 1),
                        name="conv2",
                        activation=tf.nn.relu)
                        # kernel_regularizer=l2_regularizer,
                        # bias_regularizer=l2_regularizer)

            conv_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv2")
            zoom_size = self.num_features*self.num_filters*2
        else:
            conv_vars2 = []
            zoom_size = self.num_features*self.num_filters

        flat_zoom_X = tf.reshape(zoom_X, [-1, zoom_size])

        self.logits, self.predictions, class_reg_loss = self.build_net(
            input=flat_zoom_X,
            layer_sizes=[zoom_size] + layer_sizes + [self.num_classes],
            nonlinearity=tf.nn.relu,
            scope='dnn',
            phase=self.phase)

        classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dnn")

        # Define the loss function
        cost = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=tf.reshape(tf.tile(tf.reshape(self.label, [-1]), [dynamic_num_samples]), [-1, self.num_classes]),
                name='loss_function')

        past_cost = tf.Variable(tf.zeros([]), name='past_cost')

        cost_mean = tf.reduce_mean(cost)
        cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))

        log_probability = tf.log(sample_probs + 10e-20)
        log_probability = tf.verify_tensor_all_finite(log_probability, "Undefined log probability")
        log_probability = tf.reduce_sum(log_probability, axis=1)

        # Define the loss function for the classifier
        self.loss = cost_mean
        self.policy_loss = tf.reduce_mean(tf.multiply((cost-past_cost), log_probability))
        self.reg_loss = 10*(Le + Lb) - (Lve + Lvb)

        self.pre_train_step = tf.contrib.layers.optimize_loss(
                cost_mean, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=[classifier_vars, conv_vars, conv_vars2], name='PRE_TRAIN')

        # Define the train step
        self.train_step = [
            tf.contrib.layers.optimize_loss(
                self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=[classifier_vars, conv_vars2], name='CLASSIFIER'),

            tf.contrib.layers.optimize_loss(
                self.policy_loss + self.reg_loss, global_step=self.global_step, increment_global_step=False, learning_rate=0.05*self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=policy_vars, name='REINFORCE')
        ]

        if feedback_distance == 'cosine':
            feedback_cost = tf.losses.cosine_distance(
                    labels=self.feedback,
                    predictions=tf.reshape(self.probs, [-1, self.num_features]),
                    dim=1,
                    reduction=Reduction.NONE)
        elif feedback_distance == 'mse':
            feedback_cost = tf.losses.mean_squared_error(
                    labels=self.feedback,
                    predictions=tf.reshape(self.probs, [-1, self.num_features]),
                    reduction=Reduction.NONE)
        elif feedback_distance == 'log_loss':
            feedback_cost = tf.losses.log_loss(
                    labels=self.feedback,
                    predictions=tf.reshape(self.probs, [-1, self.num_features]),
                    reduction=Reduction.NONE)

        self.feedback_train_step = tf.contrib.layers.optimize_loss(
            tf.reduce_mean(feedback_cost) + self.policy_loss + self.reg_loss, global_step=self.global_step, learning_rate=0.1*self.learning_rate, optimizer=optimizer,
            summaries=["gradients"], variables=[policy_vars], name='FEEDBACK_TRAIN')

        tf.assign(past_cost, past_cost*0.9 + 0.1*tf.reduce_mean(cost))

        self.add_logging()  # Monitor loss, accuracy and auc
        # Initialize all the local variables (used to calculate auc & accuracy)
        local_init = tf.local_variables_initializer()
        self.sess.run(local_init)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        input_images = tf.summary.image("Visualize_Input", tf.reshape(self.X, [10, frame_size, frame_size, 1]), max_outputs=10)
        prob_images = tf.summary.image("Probability_Distribution", tf.reshape(self.probs, [10, 20, 20, 1]), max_outputs=10)
        feedback_images = tf.summary.image("Feedback_Images", tf.reshape(self.feedback, [10, 20, 20, 1]), max_outputs=10)
        attention_images = tf.summary.image("Attention_Images", tf.reshape(self.attention, [10, 20, 20, 1]), max_outputs=10)

        self.merged_images = tf.summary.merge([input_images, prob_images, attention_images, feedback_images])


class Cat_Net(Model):
    """ Creates a deep learning model. """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, num_samples, frame_size, num_cat, learning_rate, feedback_distance, directory, meta=None):
        # Initialize the Model object
        super().__init__(directory, learning_rate, num_filters, num_features, frame_size, num_samples)

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

            conv = tf.layers.conv2d(
                inputs= tf.reshape(self.X, [-1, 60, 60, 1]),
                filters=self.num_filters,
                kernel_size=[5, 5],
                padding="same",
                strides=(3, 3),
                name='conv1',
                activation=tf.nn.relu)

            conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv1")
            features = tf.reshape(conv, [-1, self.num_features, self.num_filters])
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
            self.distribution = tf.contrib.distributions.Categorical(probs=tf.reshape(self.probs, [-1, self.num_features, num_cat]), allow_nan_stats=False, dtype=tf.float32)
            dynamic_num_samples = tf.floordiv(tf.shape(self.attention)[0], tf.shape(self.X)[0])

            attention_summary = tf.summary.histogram('attention', self.attention)
            # Calculate the probability of each attention vector
            sample_probs = tf.reshape(self.distribution.prob(tf.reshape(tf.cast(self.attention, tf.int32), [dynamic_num_samples, -1, self.num_features])), [-1, self.num_features])
            probability_summary = tf.summary.histogram('probability', self.probs)
            sample_probability_summary = tf.summary.histogram('sample_probability', sample_probs)

            weights = tf.linspace(0.0, 1.0, num_cat, name="weights")
            weights = tf.reshape(weights, [num_cat, -1])

            # Multiply the features by the gates defined by attention
            weighthed_probs = tf.reshape(tf.matmul(tf.reshape(self.probs, [-1, num_cat]), weights), [-1, self.num_features, 1])
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
            features = tf.reshape(tf.tile(tf.reshape(features, [-1]), [dynamic_num_samples]), [-1, self.num_features, self.num_filters])
            zoom_X = tf.multiply(features, (1/num_cat)*tf.reshape(self.attention, [-1, self.num_features, 1]), name='Zoom_X')
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
                    labels=tf.reshape(tf.tile(tf.reshape(self.label, [-1]), [dynamic_num_samples]), [-1, self.num_classes]),
                    name='loss_function')

            past_cost = tf.Variable(tf.zeros([]), name='past_cost')

            cost_mean = tf.reduce_mean(cost)
            cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))

            log_probability = tf.log(sample_probs + 10e-20)
            log_probability = tf.verify_tensor_all_finite(log_probability, "Undefined log probability")
            log_probability = tf.reduce_sum(log_probability, axis=1)

            # Define the loss function for the classifier
            self.loss = cost_mean
            self.policy_loss = tf.reduce_mean(tf.multiply((cost-past_cost), log_probability))
            self.reg_loss = (Le + Lb) - (Lve + Lvb)
            # Define the train step
            self.pre_train_step = tf.contrib.layers.optimize_loss(
                    cost_mean, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=[classifier_vars, conv_vars], name='PRE_TRAIN')

            self.train_step = [
                tf.contrib.layers.optimize_loss(
                    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=classifier_vars, name='CLASSIFIER'),

                tf.contrib.layers.optimize_loss(
                    self.policy_loss + self.reg_loss, global_step=self.global_step, increment_global_step=False, learning_rate=0.05*self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=policy_vars, name='REINFORCE')
            ]

            if feedback_distance == 'cosine':
                feedback_cost = tf.losses.cosine_distance(
                        labels=self.feedback,
                        predictions=tf.reshape(weighthed_probs, [-1, self.num_features]),
                        dim=1,
                        reduction=Reduction.NONE)
            elif feedback_distance == 'mse':
                feedback_cost = tf.losses.mean_squared_error(
                        labels=self.feedback,
                        predictions=tf.reshape(weighthed_probs, [-1, self.num_features]),
                        reduction=Reduction.NONE)
            elif feedback_distance == 'log_loss':
                feedback_cost = tf.losses.log_loss(
                        labels=self.feedback,
                        predictions=tf.reshape(weighthed_probs, [-1, self.num_features]),
                        reduction=Reduction.NONE)

            self.feedback_train_step = tf.contrib.layers.optimize_loss(
                tf.reduce_mean(feedback_cost) + self.policy_loss + self.reg_loss, global_step=self.global_step, learning_rate=0.05*self.learning_rate, optimizer=optimizer,
                summaries=["gradients"], variables=[policy_vars], name='FEEDBACK_TRAIN')

            tf.assign(past_cost, past_cost*0.9 + 0.1*tf.reduce_mean(cost))

            self.add_logging()  # Monitor loss, accuracy and auc
            # Initialize all the local variables (used to calculate auc & accuracy)
            local_init = tf.local_variables_initializer()
            self.sess.run(local_init)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            input_images = tf.summary.image("Visualize_Input", tf.reshape(self.X, [10, frame_size, frame_size, 1]), max_outputs=10)
            prob_images = tf.summary.image("Probability_Distribution", tf.reshape(weighthed_probs, [10, 20, 20, 1]), max_outputs=10)
            feedback_images = tf.summary.image("Feedback_Images", tf.reshape(self.feedback, [10, 20, 20, 1]), max_outputs=10)
            attention_images = tf.summary.image("Attention_Images", tf.reshape(self.attention, [10, 20, 20, 1]), max_outputs=10)

            self.merged_images = tf.summary.merge([input_images, prob_images, attention_images, feedback_images])


class Gumbel_Net(Model):
    """ Creates a deep learning model. """
    def __init__(self, layer_sizes, optimizer, num_filters, num_features, frame_size, num_cat, learning_rate, feedback_distance, directory, second_conv=False, initial_tau=10.0, meta=None):

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

            conv = tf.layers.conv2d(
                inputs= tf.reshape(self.X, [-1, 60, 60, 1]),
                filters=self.num_filters,
                kernel_size=[5, 5],
                padding="same",
                strides=(3, 3),
                name='conv1',
                activation=tf.nn.relu)

            conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv1")
            features = tf.reshape(conv, [-1, self.num_features, self.num_filters])
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

            # temperature
            tau_decay = tf.train.exponential_decay(learning_rate=initial_tau,
                                             global_step=self.global_step,
                                             decay_steps=100,
                                             decay_rate=0.96,
                                             staircase=True,
                                             name="temperature")

            tau_summary = tf.summary.scalar('tau', tau_decay)

            tau = tf.cond(tf.greater(self.phase, 0), lambda: tau_decay, lambda: 0.000001)

            self.distribution = self.Distribution(probs_logits, num_features, num_cat, tau)

            # Calculate the attention vector
            attention = gumbel_softmax(tf.reshape(probs_logits, [-1, self.num_features, num_cat]), tau, hard=False)

            weights = tf.linspace(0.0, 1.0, num_cat, name="weights")
            weights = tf.reshape(weights, [num_cat, -1])

            attention = tf.reshape(tf.matmul(tf.reshape(attention, [-1, num_cat]), weights), [-1, self.num_features, 1])
            attention = tf.cond(tf.equal(self.phase, 2), lambda: tf.ones_like(attention),  lambda: attention)
            attention_summary = tf.summary.histogram('attention', attention)

            # Calculate regularizers based on the sparsity of the distribution of probabilities
            batch_mean, batch_variance = tf.nn.moments(attention, axes=[0])
            example_mean, example_variance = tf.nn.moments(attention, axes=[1])
            Lb = tf.reduce_mean(tf.square(batch_mean - 0.20))
            Le = tf.reduce_mean(tf.square(example_mean - 0.20))
            Lve = tf.reduce_mean(example_variance)
            Lvb = tf.reduce_mean(batch_variance)

            Lb_summary = tf.summary.scalar('Lb', Lb)
            Le_summary = tf.summary.scalar('Le', Le)
            Lv_summary = tf.summary.scalar('Lve', Lve)
            Lv_summary = tf.summary.scalar('Lvb', Lvb)

            zoom_X = tf.multiply(features, tf.reshape(attention, [-1, self.num_features, 1]), name='Zoom_X')

            if second_conv:
                # Multiply the features by the gates defined by attention
                zoom_X = tf.layers.conv2d(
                            inputs= tf.reshape(zoom_X, [-1, 20, 20, self.num_filters]),
                            filters=self.num_filters*2,
                            kernel_size=[5, 5],
                            padding="same",
                            strides=(1, 1),
                            name="conv2",
                            activation=tf.nn.relu)
                            # kernel_regularizer=l2_regularizer,
                            # bias_regularizer=l2_regularizer)

                conv_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv2")
                zoom_size = self.num_features*self.num_filters*2
            else:
                conv_vars2 = []
                zoom_size = self.num_features*self.num_filters

            flat_zoom_X = tf.reshape(zoom_X, [-1, zoom_size])
            self.logits, self.predictions, class_reg_loss = self.build_net(
                input=flat_zoom_X,
                layer_sizes=[zoom_size] + layer_sizes + [self.num_classes],
                nonlinearity=tf.nn.relu,
                scope='dnn',
                phase=self.phase)

            classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dnn")

            # Define the loss function
            cost = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.label,
                    name='loss_function')
            cost_mean = tf.reduce_mean(cost)

            # Define the loss function for the classifier
            self.reg_loss = (Le + Lb) - (Lve + Lvb)
            self.loss = cost_mean + self.reg_loss

            self.pre_train_step = tf.contrib.layers.optimize_loss(
                    cost_mean, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                    summaries=["gradients"], variables=[classifier_vars, conv_vars, conv_vars2], name='PRE_TRAIN')

            self.train_step = tf.contrib.layers.optimize_loss(
                                self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                                summaries=["gradients"], name='TRAIN')

            if feedback_distance == 'cosine':
                feedback_cost = tf.losses.cosine_distance(
                        labels=self.feedback,
                        predictions=tf.reshape(attention, [-1, self.num_features]),
                        dim=1,
                        reduction=Reduction.NONE)
            elif feedback_distance == 'mse':
                feedback_cost = tf.losses.mean_squared_error(
                        labels=self.feedback,
                        predictions=tf.reshape(attention, [-1, self.num_features]),
                        reduction=Reduction.NONE)
            elif feedback_distance == 'log_loss':
                feedback_cost = tf.losses.log_loss(
                        labels=self.feedback,
                        predictions=tf.reshape(attention, [-1, self.num_features]),
                        reduction=Reduction.NONE)

            self.feedback_train_step = tf.contrib.layers.optimize_loss(
                                tf.reduce_mean(feedback_cost) + cost_mean + self.reg_loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=optimizer,
                                summaries=["gradients"])

            self.add_logging()  # Monitor loss, accuracy and auc
            # Initialize all the local variables (used to calculate auc & accuracy)
            local_init = tf.local_variables_initializer()
            self.sess.run(local_init)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            input_images = tf.summary.image("Visualize_Input", tf.reshape(self.X, [10, frame_size, frame_size, 1]), max_outputs=10)
            prob_images = tf.summary.image("Probability_Distribution", tf.reshape(attention, [10, 20, 20, 1]), max_outputs=10)
            feedback_images = tf.summary.image("Feedback_Images", tf.reshape(self.feedback, [10, 20, 20, 1]), max_outputs=10)
            attention_images = tf.summary.image("Attention_Images", tf.reshape(self.attention, [10, 20, 20, 1]), max_outputs=10)

            self.merged_images = tf.summary.merge([input_images, prob_images, attention_images, feedback_images])

    class Distribution:
        def __init__(self, probs_logits, num_features, num_cat, tau):
            self.probs_logits = probs_logits
            self.num_features = num_features
            self.num_cat = num_cat
            self.tau = tau

        def sample(self, num_samples=1):
            sample = gumbel_softmax(tf.reshape(self.probs_logits, [-1, self.num_cat]), self.tau, hard=False)
            return tf.reshape(tf.divide(tf.argmax(sample, axis=1), self.num_cat - 1), [-1, self.num_features])
