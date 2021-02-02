import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense


def sample_noise(shape):
    noise = tf.random.normal(shape)
    return noise


# the function used in eq.7,8
def f(x):
    return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))


class NoisyDense(Dense):

    def call(self, inputs):
        p = sample_noise([inputs.get_shape().as_list()[1], 1])
        q = sample_noise([1, self.units])
        f_p = f(p)
        f_q = f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        # w = w_mu + w_sigma*w_epsilon
        self.kernel = self.mu + tf.multiply(self.sigma, w_epsilon)

        outputs = tf.matmul(inputs, self.kernel)

        self.bias = self.bias_mu + tf.multiply(self.bias_sigma, b_epsilon)
        outputs = outputs + self.bias
        return self.activation(outputs)

    def build(self, input_shape):
        last_dim = input_shape[-1]

        self.mu_init = tf.random_uniform_initializer(minval=-1 * 1 / np.power(last_dim, 0.5),
                                                     maxval=1 * 1 / np.power(last_dim, 0.5))
        self.sigma_init = tf.constant_initializer(0.4 / np.power(last_dim, 0.5))

        self.mu = self.add_weight(
            'kernel_mu',
            shape=[last_dim, self.units],
            initializer=self.mu_init,
            trainable=True)
        self.sigma = self.add_weight(
            'kernel_sigma',
            shape=[last_dim, self.units],
            initializer=self.sigma_init,
            trainable=True)
        if self.use_bias:
            self.bias_mu = self.add_weight(
                'bias_mu',
                shape=[self.units, ],
                initializer=self.mu_init,
                trainable=True)
            self.bias_sigma = self.add_weight(
                'bias_sigma',
                shape=[self.units, ],
                initializer=self.sigma_init,
                trainable=True)
