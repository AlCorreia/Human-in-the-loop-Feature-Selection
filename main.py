import argparse
import numpy as np
import os
import sys
import tensorflow as tf
from tqdm import tqdm

from models import Bernoulli_Net, Cat_Net
from utils import *


def main(_):
    mnist = load_mnist()

    pre_training = False
    if FLAGS.model == 'Conv':
        pre_training = False
        net = Simple_Conv_Net(directory=FLAGS.dir, optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate, layer_sizes=FLAGS.arch,
            num_cat=FLAGS.num_cat)
    if FLAGS.model == 'Cat':
        pre_training = True
        net = Cat_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.num_features, frame_size=FLAGS.frame_size, num_cat=FLAGS.num_cat, learning_rate=FLAGS.learning_rate, directory=FLAGS.dir)
    elif FLAGS.model == 'Gumbel':
        pre_training = False
        net = Gumbel_Net(directory=FLAGS.dir, optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate, layer_sizes=FLAGS.arch,
            num_cat=FLAGS.num_cat, frame_size=FLAGS.frame_size)
    elif FLAGS.model == 'RL':
        pre_training = True
        net = Bernoulli_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.num_features, frame_size=FLAGS.frame_size, learning_rate=FLAGS.learning_rate, directory=FLAGS.dir)

    X_train, train_coords = convertCluttered(mnist.train.images, finalImgSize=FLAGS.frame_size)
    y_train = mnist.train.labels

    train_coords = np.array([gkern(coord[0], coord[1], kernlen=20) for coord in train_coords])

    X_test, test_coords = convertCluttered(mnist.test.images, finalImgSize=FLAGS.frame_size)
    # test_coords = np.array([gkern(coord[0], coord[1], kernlen=20) for coord in test_coords])
    y_test = mnist.test.labels

    batch_size=FLAGS.batch_size
    if pre_training:
        print("Pre-training")
        for epoch in tqdm(range(FLAGS.epochs)):
            _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
            net.evaluate(_x, _y, pre_trainining=True)
            # print(net.confusion_matrix(_x, _y))
            net.save()
            X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
            for i in range(0, len(X_train), batch_size):
                _x, _y = input_fn(X_train[i:i+batch_size], y_train[i:i+batch_size], batch_size=batch_size)
                net.pre_train(_x, _y)

    print("Training")
    for epoch in tqdm(range(FLAGS.epochs)):
        X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
        _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
        net.evaluate(_x, _y)
        # print(net.confusion_matrix(_x, _y))
        net.save()
        for i in range(0, len(X_train), batch_size):
            _x, _y, _train_coords = X_train[i:i+batch_size], y_train[i:i+batch_size], train_coords[i:i+batch_size]
            net.train(_x, _y)

    if FLAGS.model == 'RL' or FLAGS.model == 'Gumbel' or FLAGS.model == 'Cat':
        print("Feedback Training")
        for epoch in tqdm(range(FLAGS.epochs)):
            _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
            net.evaluate(_x, _y)
            # print(net.confusion_matrix(_x, _y))
            net.save()
            X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
            for i in range(0, len(X_train), batch_size):
                _x, _y, _train_coords = input_fn(X_train, y_train, train_coords, batch_size=batch_size)
                net.feedback_train(_x, _y, _train_coords)


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
        default=3,
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
        default='128,32',
        help=''' The number of neurons in each layer of the neural net classifier. '''
    )

    parser.add_argument(
        '--num_filters', '-fi',
        type=int,
        default='32',
        help=''' The number of filters in the conv net. '''
    )

    parser.add_argument(
        '--num_features', '-fe',
        type=int,
        default='400',
        help=''' The number of features used by the agent. '''
    )

    parser.add_argument(
        '--nonlinearity', '-f',
        type=str,
        default='tf.nn.relu',
        help=''' The neural net activation function. '''
    )

    parser.add_argument(
        '--frame_size', '-fr',
        type=str,
        default=60,
        help=''' The size of the cluttered MNIST image. '''
    )

    parser.add_argument(
        '--feedback_distance', '-fd',
        type=str,
        default='cosine',
        help=''' The dissimilarity measure used to evaluate the feedback. '''
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
