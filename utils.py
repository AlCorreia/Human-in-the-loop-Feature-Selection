import numpy as np
import random
from scipy.stats import multivariate_normal
import tensorflow as tf


def input_fn(X, y, z=None, batch_size=1000, output_size=10):
    """ Returns a batch of examples randomly selected from a numpy 2D-array.

        Parameters
        ----------
        X : numpy 2D-array
            Matrix containing all the features used in the model.
        y : numpy array
            The targets of the model.
        batch_size : int
            The number of examples treated in a iteration.

        Returns
        ----------
        state : numpy 2D-array, batch_size X number_of_features
            Matrix with all the features for the randomly selected examples
        state : numpy array, batch_size
            Array with the corresponding target for the examples

    """
    # Returns a random state for each episode.
    index = [np.random.randint(1, len(X)) for i in range(batch_size)]
    state = X[index].reshape(batch_size, -1)
    label = y[index].reshape(batch_size, output_size)
    if z is None:
        return state, label
    else:
        return state, label, z[index].reshape(batch_size, -1)


def shuffle_in_unison(a, b, c=None):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(a.shape[0])
    if c is not None:
        shuffled_c = np.empty(c.shape, dtype=c.dtype)
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
            shuffled_c[new_index] = c[old_index]
        return shuffled_a, shuffled_b, shuffled_c
    else:
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b


def sample_gumbel(shape, eps=1e-20):
  """ Sample from Gumbel(0, 1) """
  U = tf.random_uniform(shape, minval=0, maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
  """ Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y
