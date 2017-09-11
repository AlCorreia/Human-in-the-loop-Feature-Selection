import argparse
import math
import numpy as np
import os
import sys
import tensorflow as tf
from tqdm import tqdm

from models import Baseline, Bernoulli_Net, Cat_Net, Gumbel_Net, Raw_Bernoulli_Net, Raw_Gumbel_Net
from utils import *


def str2bool(v):
    """ Converts a string to a boolean value. """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(_):
    # Load MNIST data
    mnist = load_mnist()
    pre_training = FLAGS.pre_train

    # Define the deep learning model
    if FLAGS.model == 'Base':
        pre_training = False
        kernlen = int(FLAGS.frame_size/2)
        net = Baseline(directory=FLAGS.dir, optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate, layer_sizes=FLAGS.arch, num_features=FLAGS.num_features, num_filters=FLAGS.num_filters, frame_size=FLAGS.frame_size)
    if FLAGS.model == 'Cat':
        kernlen = int(FLAGS.frame_size/2)
        net = Cat_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.num_features, num_samples=FLAGS.num_samples, frame_size=FLAGS.frame_size, num_cat=FLAGS.num_cat, learning_rate=FLAGS.learning_rate, feedback_distance=FLAGS.feedback_distance, directory=FLAGS.dir)
    elif FLAGS.model == 'Gumbel':
        kernlen = int(FLAGS.frame_size/2)
        net = Gumbel_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.num_features, frame_size=FLAGS.frame_size, num_cat=FLAGS.num_cat, learning_rate=FLAGS.learning_rate, feedback_distance=FLAGS.feedback_distance, directory=FLAGS.dir, second_conv=FLAGS.second_conv, initial_tau=FLAGS.initial_tau, tau_decay=FLAGS.tau_decay, reg=FLAGS.reg)
    elif FLAGS.model == 'RawG':
        pre_training = False
        kernlen = 60
        net = Raw_Gumbel_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.frame_size**2, frame_size=FLAGS.frame_size, num_cat=FLAGS.num_cat, learning_rate=FLAGS.learning_rate, feedback_distance=FLAGS.feedback_distance, directory=FLAGS.dir, second_conv=FLAGS.second_conv, initial_tau=FLAGS.initial_tau, meta=None)
    elif FLAGS.model == 'RL':
        kernlen = int(FLAGS.frame_size/2)
        net = Bernoulli_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.num_features, num_samples=FLAGS.num_samples, frame_size=FLAGS.frame_size, learning_rate=FLAGS.learning_rate, feedback_distance=FLAGS.feedback_distance, directory=FLAGS.dir, second_conv=FLAGS.second_conv)
    elif FLAGS.model == 'RawB':
        pre_training = True
        kernlen = 60
        net = Raw_Bernoulli_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.frame_size**2, num_samples=FLAGS.num_samples, frame_size=FLAGS.frame_size, learning_rate=FLAGS.learning_rate, feedback_distance=FLAGS.feedback_distance, directory=FLAGS.dir, second_conv=FLAGS.second_conv)

    X_train, train_coords = convertCluttered(mnist.train.images, finalImgSize=FLAGS.frame_size, number_patches=FLAGS.number_patches)
    y_train = mnist.train.labels

    train_coords = np.array([gkern(coord[0], coord[1], kernlen=kernlen) for coord in train_coords])

    X_test, test_coords = convertCluttered(mnist.test.images, finalImgSize=FLAGS.frame_size, number_patches=FLAGS.number_patches)
    # test_coords = np.array([gkern(coord[0], coord[1], kernlen=20) for coord in test_coords])
    y_test = mnist.test.labels

    batch_size = FLAGS.batch_size
    if pre_training:
        print("Pre-training")
        for epoch in tqdm(range(FLAGS.epochs)):
            _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
            net.evaluate(_x, _y, pre_trainining=True)
            X_train, train_coords = convertCluttered(mnist.train.images, finalImgSize=FLAGS.frame_size, number_patches=FLAGS.number_patches)
            y_train = mnist.train.labels
            # print(net.confusion_matrix(_x, _y))
            net.save()
            X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
            for i in range(0, len(X_train), batch_size):
                _x, _y = input_fn(X_train[i:i+batch_size], y_train[i:i+batch_size], batch_size=batch_size)
                net.pre_train(_x, _y, dropout=0.8)

    print("Training")
    for epoch in tqdm(range(FLAGS.epochs)):
        X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
        _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
        net.evaluate(_x, _y)
        X_train, train_coords = convertCluttered(mnist.train.images, finalImgSize=FLAGS.frame_size, number_patches=FLAGS.number_patches)
        y_train = mnist.train.labels
        # print(net.confusion_matrix(_x, _y))
        net.save()
        for i in range(0, len(X_train), batch_size):
            _x, _y = X_train[i:i+batch_size], y_train[i:i+batch_size]
            net.train(_x, _y, dropout=FLAGS.dropout)

    if FLAGS.model == 'RL' or FLAGS.model == 'Gumbel' or FLAGS.model == 'Cat' or FLAGS.model == 'RawB' or FLAGS.model == 'RawG':
        print("Feedback Training")
        for epoch in tqdm(range(FLAGS.epochs)):
            _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
            net.evaluate(_x, _y)
            X_train, train_coords = convertCluttered(mnist.train.images, finalImgSize=FLAGS.frame_size, number_patches=FLAGS.number_patches)
            y_train = mnist.train.labels
            train_coords = np.array([gkern(coord[0], coord[1], kernlen=kernlen) for coord in train_coords])
            # print(net.confusion_matrix(_x, _y))
            net.save()
            X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
            for i in range(0, len(X_train), batch_size):
                _x, _y, _train_coords = input_fn(X_train, y_train, train_coords, batch_size=batch_size)
                net.feedback_train(_x, _y, _train_coords, dropout=FLAGS.dropout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='RL',
        help='''Categorical variables representation:
                binary for binary encoding, embedded for word embedding. '''
    )

    parser.add_argument(
        '--optimizer', '-o',
        type=str,
        default='Adagrad',
        help='Optimizer.'
    )

    parser.add_argument(
        '--learning_rate', '-l',
        type=float,
        default=0.1,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=25,
        help='The number of iterations to run'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=1000,
        help='The batch size'
    )

    parser.add_argument(
        '--num_cat', '-nc',
        type=int,
        default=2,
        help='The number of categories for categorical distributions. '
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='./model',
        help='''Categorical variables representation:
                binary for binary encoding, embedded for word embedding.'''
    )

    parser.add_argument(
        '--arch', '-A',
        type=str,
        default='256',
        help=''' The number of neurons in each layer of the neural net classifier. '''
    )

    parser.add_argument(
        '--num_filters', '-fi',
        type=int,
        default=8,
        help=''' The number of filters in the conv net. '''
    )

    parser.add_argument(
        '--num_features', '-fe',
        type=int,
        default=900,
        help=''' The number of features used by the agent. '''
    )

    parser.add_argument(
        '--num_samples', '-sa',
        type=int,
        default=1,
        help=''' The number of samples per example. '''
    )

    parser.add_argument(
        '--nonlinearity', '-f',
        type=str,
        default='tf.nn.relu',
        help=''' The neural net activation function. '''
    )

    parser.add_argument(
        '--frame_size', '-fr',
        type=int,
        default=60,
        help=''' The size of the cluttered MNIST image. '''
    )

    parser.add_argument(
        '--feedback_distance', '-fd',
        type=str,
        default='cosine',
        help=''' The dissimilarity measure used to evaluate the feedback. '''
    )

    parser.add_argument(
        '--second_conv', '-sc',
        type=str2bool,
        default='false',
        help=''' Wheter to add a second convolutional layer. '''
    )

    parser.add_argument(
        '--dropout', '-dr',
        type=float,
        default=1.0,
        help=''' The probability of keeping a neuron active. '''
    )

    parser.add_argument(
        '--initial_tau', '-t',
        type=float,
        default=10.0,
        help=''' The initial temperature vaule for the PD model. '''
    )

    parser.add_argument(
        '--tau_decay', '-td',
        type=str2bool,
        default='true',
        help=''' Whether to decay tau or keep it constant. '''
    )

    parser.add_argument(
        '--reg', '-r',
        type=float,
        default=1.0,
        help=''' Trade-off between classification and regularization cost. '''
    )

    parser.add_argument(
        '--pre_train', '-p',
        type=str2bool,
        default='true',
        help=''' Whether the model should be pre trained or not. '''
    )

    parser.add_argument(
        '--number_patches', '-np',
        type=int,
        default=10,
        help=''' The number of noise patches to be added to the model. '''
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.arch = [int(item) for item in FLAGS.arch.split(',')]
    if not os.path.exists(FLAGS.dir):
        os.makedirs(FLAGS.dir)
    else:
        raise ValueError('This model\'s name has already been used.')
    with open(FLAGS.dir + '/config', 'w') as f:
        f.write(str(vars(FLAGS)))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
