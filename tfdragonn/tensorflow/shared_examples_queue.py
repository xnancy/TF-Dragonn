from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from collections import defaultdict

import tensorflow as tf

"""
A shared examples queue for multiple datasets.
"""


def get_shared_batch(readers, batch_size=128, shuffle=True, capacity=5000,
                     name='shared-examples-queue'):
    """Generate a batch of examples from all datasets given a dict of their readers.
    Args:
        readers: a dictionary of readers, returned by `get_readers`
        batch_size: the size of batched tensors to return
        shuffle: whether to shuffle the returned batches
        capacity: the size of the buffer used for batching
        name: the namespace used
    Returns:
        batch: a dictionary of tensors with first dimension batch_size

    For consitency, each reader must have the same fields.
    """
    reader_keys = list(readers.keys())  # keep order consistent
    data_keys = set()
    for reader in readers.values():
        data_keys = data_keys.union(reader.names)
    data_keys = list(data_keys)

    dtypes_and_shapes = OrderedDict
    for r_name, r in readers.items():
        for name, dtype, shape in zip(r.names, r.dtypes, r.shapes):
            dtypes_and_shapes[name] = (dtype, shape)

    with tf.variable_scope(name):
        dequeues = {k: v.dequeue() for k, v in readers.items()}
        tensors_to_enqueue = {}
        for tensor_name in data_keys:
            values = [dequeues[k][tensor_name] for k in reader_keys]
            tensors_to_enqueue[tensor_name] = tf.stack(values)

        # Add the dataset names so we can evaluate per-dataset
        tensors_to_enqueue['dataset'] = tf.convert_to_tensor(reader_keys)

        if shuffle:
            min_after_dequeue = int(0.5 * capacity)
            outputs = tf.train.shuffle_batch(tensors_to_enqueue, batch_size, capacity=capacity,
                                             min_after_dequeue=min_after_dequeue, num_threads=4,
                                             enqueue_many=True)

        else:
            outputs = tf.train.batch(tensors_to_enqueue, batch_size, num_threads=4,
                                     capacity=capacity, enqueue_many=True)

        return outputs
