from math import e, pi

import numpy as np
import tensorflow as tf


def compute_log_p_gaussian(x, means_and_log_stds):
    means, log_stds = tf.split(means_and_log_stds, 2, axis=1)
    std = tf.exp(log_stds)
    return (
        -0.5
        * tf.reduce_sum(tf.math.square((tf.cast(x, tf.float32) - means) / std), axis=1)
        - 0.5 * tf.math.log(2.0 * pi) * tf.cast(tf.shape(x)[1], tf.float32)
        - tf.reduce_sum(log_stds, axis=1)
    )


def compute_log_p_discrete(action_one_hot, action_prob):
    return -tf.keras.losses.categorical_crossentropy(
        action_one_hot, action_prob, from_logits=False
    )


def compute_entropy_discrete(action_prob):
    return tf.reduce_mean(
        -tf.reduce_sum(
            (action_prob * tf.math.log(action_prob + 1e-10)),
            axis=1,
        )
    )


def compute_entropy_gaussian(means_and_log_stds):
    means, log_stds = tf.split(means_and_log_stds, 2, axis=1)
    return tf.reduce_sum(log_stds + 0.5 * tf.math.log(2.0 * pi * e), axis=1)


def one_hot_encode(x, max_x):
    x = x.flatten()
    x_one_hot = np.zeros((x.size, max_x))
    x_one_hot[np.arange(x.size), x] = 1
    return x_one_hot
