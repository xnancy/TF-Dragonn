from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


DEFAULT_BIAS_VALUE = 0.1
DEFAULT_WEIGHT_STDEV = 0.1


def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TF kernel shape: (..., input_depth, depth)
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    else:
        raise ArithmeticError(
            'Unrecognized weight dimensionality: {}'.format(len(shape)))
        # no specific assumptions
        # fan_in = np.sqrt(np.prod(shape))
        # fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def get_he_normal_scale(shape):
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(2.0 / fan_in)
    return scale


def get_glorot_uniform_scale(shape):
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(6.0 / (fan_in + fan_out))
    return scale


def glorot_uniform_initializer(dtype=tf.float32, **kwargs):
    def _glorot_uniform_initializer(shape, dtype=dtype, **kwargs):
        scale = get_glorot_uniform_scale(shape)
        return tf.random_normal(shape, 0.0, scale, dtype=dtype, **kwargs)
    return _glorot_uniform_initializer


def glorot_uniform(shape, dtype=tf.float32, name=None):
    scale = get_glorot_uniform_scale(shape)
    return tf.random_uniform(scale, dtype=dtype, name=name)


def he_normal_initializer(dtype=tf.float32):
    def _he_normal_initializer(shape, dtype=dtype, **kwargs):
        scale = get_he_normal_scale(shape)
        return tf.random_normal(shape, 0.0, scale, dtype=dtype, **kwargs)
    return _he_normal_initializer


def he_normal(shape, dtype=tf.float32):
    scale = get_he_normal_scale(shape)
    return tf.random_normal(shape, 0.0, scale, dtype=dtype)


def weight_variable_initializer(stdev=DEFAULT_WEIGHT_STDEV, dtype=tf.float32):
    return tf.truncated_normal_initializer(stddev=stdev, dtype=dtype)


def weight_variable(shape, stdev=DEFAULT_WEIGHT_STDEV, dtype=tf.float32):
    return tf.truncated_normal(shape, stddev=stdev, dtype=dtype)


def bias_variable_initializer(value=DEFAULT_BIAS_VALUE, dtype=tf.float32):
    return tf.constant_initializer(value, dtype=dtype)


def bias_variable(shape, dtype=tf.float32, value=DEFAULT_BIAS_VALUE):
    return tf.constant(value, dtype=dtype, shape=shape)
