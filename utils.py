import numpy as np
import random
from scipy.stats import multivariate_normal
import tensorflow as tf


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data", one_hot=True)


def input_fn(X, y, z=None, batch_size=1000, num_classes=10):
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
    label = y[index].reshape(batch_size, num_classes)
    if z is None:
        return state, label
    else:
        return state, label, z[index].reshape(batch_size, -1)


def shuffle_in_unison(a, b, c=None):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
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


def convertTranslated(images, finalImgSize, initImgSize=28, batch_size=None):
    """ Returns translated MNIST images.

        Parameters
        ----------
        images : numpy 2D-array
            The set of images to be converted.
        finalImgSize : int
            The larger frame where the digit is positioned.
        initImgSize : int
            The size of the original image (28 for MNIST).
        batch_size: int

        Returns
        ----------
        cluttered_images : numpy 2D-array
            The augmented images with clutter
        imgCoord : numpy array
            The coordinates of the center of the digit in the new images
    """

    if batch_size is None:
        batch_size = len(images)
    size_diff = finalImgSize - initImgSize
    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgCoord = np.zeros([batch_size, 2])
    for k in range(batch_size):
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        # generate and save random coordinates
        randX = random.randint(0, size_diff)
        randY = random.randint(0, size_diff)
        imgCoord[k,:] = np.array([(randX + initImgSize/2)/finalImgSize, (randY + initImgSize/2)/finalImgSize])
        # padding
        image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
        newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

    return newimages, imgCoord


def convertCluttered(original_images, finalImgSize, initImgSize=28, number_patches=4, clutter_size=8, batch_size=None):
    """ Returns cluttered MNIST images.

        Parameters
        ----------
        original_images : numpy 2D-array
            The set of images to be converted.
        finalImgSize : int
            The larger frame where the digit is positioned.
        initImgSize : int
            The size of the original image (28 for MNIST).
        number_patches: int
            The number of patches from other images added
        clutter_size: int
            The size of the noise patches
        batch_size: int

        Returns
        ----------
        cluttered_images : numpy 2D-array
            The augmented images with clutter
        imgCoord : numpy array
            The coordinates of the center of the digit in the new images
    """

    images, imgCoord = convertTranslated(original_images, batch_size=batch_size, initImgSize=initImgSize, finalImgSize=finalImgSize)
    if batch_size is None:
        batch_size = len(images)
    size_diff = finalImgSize - clutter_size
    clutter_size_diff = initImgSize - clutter_size/2
    cluttered_images = np.zeros([batch_size, finalImgSize*finalImgSize])

    for k in range(batch_size):
        image = images[k, :]
        image = np.reshape(image, (finalImgSize, finalImgSize))
        cluttered_image = image
        for l in range(number_patches):

            original_image = original_images[random.randint(0, batch_size-1), :]
            original_image = np.reshape(original_image, (initImgSize, initImgSize))

            # generate and save random coordinates
            clutterX = random.randint(clutter_size/2, clutter_size_diff)
            clutterY = random.randint(clutter_size/2, clutter_size_diff)
            diff = np.int(clutter_size/2)
            clutter = original_image[clutterX-diff: clutterX+diff, clutterY-diff: clutterY+diff]
            # generate and save random coordinates
            randX = random.randint(0, size_diff)
            randY = random.randint(0, size_diff)
            # padding
            clutter = np.lib.pad(clutter, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
            cluttered_image = np.clip(cluttered_image + clutter, a_min=0, a_max=1)
        cluttered_images[k, :] = np.reshape(cluttered_image, (finalImgSize*finalImgSize))

    return cluttered_images, imgCoord


def gkern(mx, my, kernlen):
    """ Returns a 2D Gaussian kernel array. """
    x, y = np.mgrid[0:kernlen:1, 0:kernlen:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal([mx*kernlen, my*kernlen], [[3.0, 0.0], [0.0, 3.0]])
    array = rv.pdf(pos).reshape([kernlen*kernlen])
    array = array/max(array)

    return array


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
