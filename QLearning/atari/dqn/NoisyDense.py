from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops





class NoisyDense(Dense):

    def build(self, input_shape):

        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense` layer with non-floating point "
                "dtype %s" % (dtype,)
            )
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.mu_init = tf.random_uniform_initializer(
            minval=-1 * 1 / np.power(last_dim, 0.5),
            maxval=1 * 1 / np.power(last_dim, 0.5),
        )
        self.sigma_init = tf.constant_initializer(0.4 / np.power(last_dim, 0.5))

        self.mu = self.add_weight(
            "kernel_mu",
            shape=[last_dim, self.units],
            initializer=self.mu_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.sigma = self.add_weight(
            "kernel_sigma",
            shape=[last_dim, self.units],
            initializer=self.sigma_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias_mu = self.add_weight(
                "bias_mu",
                shape=[
                    self.units,
                ],
                initializer=self.mu_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
            self.bias_sigma = self.add_weight(
                "bias_sigma",
                shape=[
                    self.units,
                ],
                initializer=self.sigma_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def sample_noise(self, shape):
        noise = tf.random.normal(shape)
        return noise

    # the function used in eq.7,8
    def f(self, x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        rank = len(inputs.shape)



        # Initializer of \mu and \sigma

        # Sample noise from gaussian
        p = self.sample_noise([inputs.get_shape().as_list()[1], 1])
        q = self.sample_noise([1, self.units])
        f_p = self.f(p)
        f_q = self.f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        # w = w_mu + w_sigma*w_epsilon
        self.kernel = self.mu + tf.multiply(self.sigma, w_epsilon)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            self.bias = self.bias_mu + tf.multiply(self.bias_sigma, b_epsilon)
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
