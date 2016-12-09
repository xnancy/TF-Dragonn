from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

"""
A queue for storing intervals and optionally labels.
"""

_DEFAULT_BUFFER_CAPACITY = 10000


def examples_queue(intervals, data, labels=None, name='examples-queue'):
    """Create an examples queue to store extracted examples.

    Args:
        intervals: a dict of arrays with first dimension N. Either:
            1) intervals encoded as 'chrom': (string), 'start': (int), and 'stop': (int), or
            2) intervals encoded as 'bed3': (string) TSV entries
        data: a dict of tensors with first dimension N.
        labels: (optional) a tensor with dimensions N X T.
        name: (optional) string, name for this queue
    Returns:
        a queue reference
    """
    tensors_to_enqueue = OrderedDict()

    for k, v in intervals.items():
        tensors_to_enqueue['intervals/{}'.format(k)] = v

    for k, v in data.items():
        tensors_to_enqueue['data/{}'.format(k)] = v

    if labels is not None:
        tensors_to_enqueue['labels'] = labels

    shapes = [t.get_shape()[1:] for t in tensors_to_enqueue.values()]
    dtypes = [t.dtype for t in tensors_to_enqueue.values()]
    names = list(tensors_to_enqueue.keys())

    with tf.variable_scope(name):
        queue = tf.FIFOQueue(capacity=_DEFAULT_BUFFER_CAPACITY, dtypes=dtypes, shapes=shapes,
                             names=names, name='examples-queue')
        enqueue_op = queue.enqueue_many(tensors_to_enqueue)
        queue_runner = tf.train.QueueRunner(
            queue=queue, enqueue_ops=[enqueue_op], close_op=queue.close(),
            cancel_op=queue.close(cancel_pending_enqueues=True))
        tf.train.add_queue_runner(queue_runner, tf.GraphKeys.QUEUE_RUNNERS)

    return queue
