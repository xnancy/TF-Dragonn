from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""
A queue for storing intervals and labels.
"""

_DEFAULT_BUFFER_CAPACITY = 10000


def interval_queue(intervals, labels=None, dequeue_size=128, name='interval-queue'):
    """Create an interval queue.

    Args:
        intervals: a dict of numpy arrays, with the schema:
            'chrom': (string), 'start': (int), 'end' (int)
        labels: (optional) a dict of numpy arrays of size N, where N == len(intervals['chrom'])
        name: (optional) string, name for this queue
    Returns:
        a dictionary of tensors with keys [chrom, start, end, labels], where each
        tensor has shape equal to dequeue_size. labels is a dictionary of tensors, where
        the keys are the same as in the provided labels dict.
    """
    n_exs = next(labels.values()).shape[0]
    for k in ['chrom', 'start', 'end']:
        if k not in intervals:
            raise IOError('intervals must contain {}'.format(k))
        elif intervals[k].shape[0] != n_exs:
            raise IOError('intervals shape does not match labels shape: {} vs {}'.format(
                intervals[k].shape[0], n_exs))

    with tf.variable_scope(name):
        chrom_queue = tf.train.string_input_producer(
            k['chrom'], shuffle=False, capacity=_DEFAULT_BUFFER_CAPACITY, name='chrom-buffer')
        start_queue = tf.train.input_producer(
            k['start'], shuffle=False, capacity=_DEFAULT_BUFFER_CAPACITY, name='start-buffer')
        end_queue = tf.train.input_producer(
            k['end'], shuffle=False, capacity=_DEFAULT_BUFFER_CAPACITY, name='end-buffer')

        outputs = {
            'chrom': chrom_queue.dequeue_many(dequeue_size),
            'start': start_queue.dequeue_many(dequeue_size),
            'end': end_queue.dequeue_many(dequeue_size),
        }

        if labels:
            labels_dequeues = {}
            for label_name, label in labels.items():
                label_queue = tf.input_producer(
                    label, shuffle=False, capacity=_DEFAULT_BUFFER_CAPACITY, name=label_name)
                labels_dequeues[label_name] = label_queue.dequeue_many(
                    dequeue_size)
            outputs['labels'] = labels_dequeues

        return outputs
