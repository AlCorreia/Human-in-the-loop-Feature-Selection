import argparse
import math
import numpy as np
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sys
import tensorflow as tf
from tqdm import tqdm

from models import Baseline, Bernoulli_Net, Cat_Net, Gumbel_Net
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
    # Load data
    amazon_data = sio.loadmat('/Users/Alvinho/knowledge-elicitation-for-linear-regression/DATA_amazon/amazon_data.mat')
    X = amazon_data['X_all']
    y = amazon_data['Y_all']
    words = amazon_data['keywords']
    FLAGS.num_features = X.shape[1]

    if FLAGS.target_type == 'class':
        FLAGS.n_out = 4
        enc = OneHotEncoder()
        y = enc.fit_transform(y).toarray()
    elif FLAGS.target_type == 'regres':
        FLAGS.n_out = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    pre_training = False # FLAGS.pre_train

    # Define the deep learning model
    if FLAGS.model == 'Base':
        pre_training = False
        net = Baseline(num_features=FLAGS.num_features,
                       layer_sizes=FLAGS.arch,
                       output_size=FLAGS.n_out,
                       directory=FLAGS.dir,
                       optimizer=FLAGS.optimizer,
                       learning_rate=FLAGS.learning_rate,
                       target=FLAGS.target_type)

    if FLAGS.model == 'Cat':
        net = Cat_Net(num_features=FLAGS.num_features,
                      layer_sizes=FLAGS.arch,
                      output_size=FLAGS.n_out,
                      num_samples=FLAGS.num_samples,
                      feedback_distance=FLAGS.feedback_distance,
                      optimizer=FLAGS.optimizer,
                      learning_rate=FLAGS.learning_rate,
                      directory=FLAGS.dir,
                      num_cat=FLAGS.num_cat,
                      target=FLAGS.target_type)

    elif FLAGS.model == 'Gumbel':
        net = Gumbel_Net(num_features=FLAGS.num_features,
                         layer_sizes=FLAGS.arch,
                         output_size=FLAGS.n_out,
                         feedback_distance=FLAGS.feedback_distance,
                         optimizer=FLAGS.optimizer,
                         learning_rate=FLAGS.learning_rate,
                         directory=FLAGS.dir,
                         num_cat=FLAGS.num_cat,
                         initial_tau=FLAGS.initial_tau,
                         tau_decay=FLAGS.tau_decay,
                         reg=FLAGS.reg,
                         target=FLAGS.target_type)

    elif FLAGS.model == 'RL':
        net = Bernoulli_Net(num_features=FLAGS.num_features,
                            layer_sizes=FLAGS.arch,
                            output_size=FLAGS.n_out,
                            num_samples=FLAGS.num_samples,
                            feedback_distance=FLAGS.feedback_distance,
                            optimizer=FLAGS.optimizer,
                            learning_rate=FLAGS.learning_rate,
                            directory=FLAGS.dir,
                            target=FLAGS.target_type)

    batch_size = FLAGS.batch_size
    if pre_training:
        print("Pre-training")
        for epoch in tqdm(range(FLAGS.epochs)):
            _x, _y = input_fn(X_test, y_test, batch_size=batch_size, output_size=FLAGS.n_out)
            net.evaluate(_x, _y, pre_trainining=True)
            # TODO: define X_train, y_train

            net.save()
            X_train, y_train = shuffle_in_unison(X_train, y_train)
            for i in range(0, len(X_train), batch_size):
                _x, _y = input_fn(X_train[i:i+batch_size], y_train[i:i+batch_size], batch_size=batch_size, output_size=FLAGS.n_out)
                net.pre_train(_x, _y, dropout=0.8)

    print("Training")
    for epoch in tqdm(range(FLAGS.epochs)):
        X_train, y_train = shuffle_in_unison(X_train, y_train)
        _x, _y = input_fn(X_test, y_test, batch_size=batch_size, output_size=FLAGS.n_out)
        net.evaluate(_x, _y)
        net.save()
        for i in range(0, len(X_train), batch_size):
            _x, _y = X_train[i:i+batch_size], y_train[i:i+batch_size]
            net.train(_x, _y, dropout=FLAGS.dropout)

    # if FLAGS.model == 'RL' or FLAGS.model == 'Gumbel' or FLAGS.model == 'Cat' or FLAGS.model == 'RawB' or FLAGS.model == 'RawG':
    #     print("Feedback Training")
    #     for epoch in tqdm(range(FLAGS.epochs)):
    #         _x, _y = input_fn(X_test, y_test, batch_size=batch_size, output_size=FLAGS.n_out)
    #         net.evaluate(_x, _y)
    #         # print(net.confusion_matrix(_x, _y))
    #         net.save()
    #         X_train, y_train = shuffle_in_unison(X_train, y_train)
    #         for i in range(0, len(X_train), batch_size):
    #             _x, _y = input_fn(X_train, y_train, batch_size=batch_size, output_size=FLAGS.n_out)
    #             net.feedback_train(_x, _y, dropout=FLAGS.dropout)


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
        default=100,
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
        '--feedback_distance', '-fd',
        type=str,
        default='cosine',
        help=''' The dissimilarity measure used to evaluate the feedback. '''
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
        '--target_type', '-tt',
        type=str,
        default='regres',
        help=''' 'regres' for regression or 'class' for classification. '''
    )

    parser.add_argument(
        '--n_out', '-no',
        type=int,
        default=1,
        help=''' Dimension of the output. For 'regres' it is set to 1 by default . '''
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
