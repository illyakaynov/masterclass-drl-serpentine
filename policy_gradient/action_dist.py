import tensorflow as tf

from math import e, pi

class ActionDistribution:
    def __init__(self, inputs):
        self.inputs = inputs

    def sample(self):
        """
        Sample action from a current distribution
        :return:
        """
        raise NotImplemented

    def deterministic_sample(self):
        """
        Sample greedily from a current distribution
        :return:
        """
        raise NotImplemented

    def log_p(self, x):
        """
        Calculate log probability ln(pi) of the action under current distribution
        :param x:
        :return:
        """
        raise NotImplemented

    def entropy(self):
        """
        Calculate the entropy of the current distribution
        :return:
        """
        raise NotImplemented


class CategoricalDistribution(ActionDistribution):

    def log_p(self, x):
        if len(tf.shape(x)) >= 2:
            x = tf.squeeze(x)
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32)
        )

    def entropy(self):
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=1)

    def sample(self):
        return tf.squeeze(tf.random.categorical(self.inputs, 1), axis=1)

    def deterministic_sample(self):
        return tf.math.argmax(self.inputs, axis=1)


class GaussianDistribution(ActionDistribution):
    def __init__(self, inputs):
        mean, log_std = tf.split(inputs, 2, axis=1)
        self.mean = mean
        self.log_std = log_std
        self.std = tf.exp(log_std)

    def log_p(self, x):
        return -0.5 * tf.reduce_sum(
            tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
            axis=1
        ) - 0.5 * tf.math.log(2.0 * pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
            tf.reduce_sum(self.log_std, axis=1)

    def entropy(self):
        return tf.reduce_sum(
            self.log_std + .5 * tf.math.log(2.0 * pi * e), axis=1)

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    def deterministic_sample(self):
        return self.mean