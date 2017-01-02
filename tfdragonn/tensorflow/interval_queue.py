from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""
A queue for storing intervals and labels.
"""

_DEFAULT_CAPACITY = 10000


def interval_queue(intervals, labels=None, dequeue_size=128, name='interval-queue'):
    """Create an interval queue.

    The intervals are provided as a dictionary of chrom, start, and end 1D numpy arrays. The labels
    (which are optional) can be provided as a 2D numpy array. A dictionary of tensors with length
    equal to `dequeue_size` is returned.

    Args:
        intervals (dict): dictionary of numpy arrays, with the format:
            {'chrom': np.array(string), 'start': np.array(int), 'end': np.array(int)}
        labels: (optional) a numpy array of size N x T where N is the length of intervals and
            T is the number of tasks
        name: (optional) string, name for this queue

    Returns:
        a dictionary of tensors with keys [chrom, start, end, labels], where each
        tensor has shape equal to dequeue_size. Labels is T-dimensional (T = number of tasks)
    """
    n_exs = intervals['chrom'].shape[0]
    for k in ['chrom', 'start', 'end']:
        if k not in intervals:
            raise ValueError('intervals must contain {}'.format(k))
        elif intervals[k].shape[0] != n_exs:
            raise ValueError('intervals shape does not match labels shape: {} vs {}'.format(
                intervals[k].shape[0], n_exs))

    with tf.variable_scope(name):
        chrom_queue = tf.train.string_input_producer(
            tf.convert_to_tensor(intervals['chrom']), shuffle=False,
            capacity=_DEFAULT_CAPACITY, name='chrom-buffer')
        start_queue = tf.train.input_producer(
            tf.convert_to_tensor(intervals['start']), shuffle=False,
            capacity=_DEFAULT_CAPACITY, name='start-buffer')
        end_queue = tf.train.input_producer(
            tf.convert_to_tensor(intervals['end']), shuffle=False,
            capacity=_DEFAULT_CAPACITY, name='end-buffer')

        outputs = {
            'chrom': chrom_queue.dequeue_many(dequeue_size),
            'start': start_queue.dequeue_many(dequeue_size),
            'end': end_queue.dequeue_many(dequeue_size),
        }

        if labels is not None:
            label_queue = tf.train.input_producer(
                tf.convert_to_tensor(labels), shuffle=False, capacity=_DEFAULT_CAPACITY,
                name='labels-buffer')
            labels_dequeue = label_queue.dequeue_many(dequeue_size)

            outputs['labels'] = labels_dequeue

        return outputs
