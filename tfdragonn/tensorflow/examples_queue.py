from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

"""
A queue for storing intervals and labels.
"""

_DEFAULT_BUFFER_CAPACITY = 10000


def examples_queue(intervals, data, labels, name='examples-queue'):
    """Create an examples queue.

    Args:
        intervals: a dict of tensors with first dimension N. Either:
            1) intervals encoded as 'chrom': (string), 'start': (int), and 'stop': (int), or
            2) intervals encoded as 'bed3': (string) TSV entries
        data: a dict of tensors with first dimension N.
        labels: a dict of tensors, each with first dimension N.
        name: (optional) string, name for this queue
    Returns:
        a queue reference
    """
    tensors_to_enqueue = OrderedDict

    for k, v in intervals.items():
        assert k not in tensors_to_enqueue
        tensors_to_enqueue[k] = v

    for k, v in data.items():
        assert k not in tensors_to_enqueue
        tensors_to_enqueue[k] = v

    for k, v in labels.items():
        assert k not in tensors_to_enqueue
        tensors_to_enqueue[k] = v

    shapes = [t.get_shape()[1:] for t in tensors_to_enqueue.values()]
    dtypes = [t.dtype for t in tensors_to_enqueue.values()]
    names = list(tensors_to_enqueue.keys())

    with tf.variable_scope(name):
        queue = tf.FIFOQueue(capacity=_DEFAULT_BUFFER_CAPACITY, dtypes=dtypes, shapes=shapes,
                             names=names, name='examples-queue')

    return queue
