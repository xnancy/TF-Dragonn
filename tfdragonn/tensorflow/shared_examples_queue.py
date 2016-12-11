from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SharedExamplesQueue(object):
    """A container for a shared examples queue and metadata."""

    def __init__(self, readers, task_names, batch_size=128, shuffle=True, capacity=50000,
                 name='shared-examples-queue'):
        """Construct a shared examples queue for a dict of readers.

        Args:
            readers: a dictionary of readers, returned by `get_readers`
            batch_size: the size of batched tensors to return
            shuffle: whether to shuffle the returned batches
            capacity: the size of the buffer used for batching
            name: the namespace used

        For consitency, each reader must have the same fields.
        """
        self.name = name
        # keep order consistent
        self.reader_keys = sorted(list(readers.keys()))
        data_keys = set()
        for reader in readers.values():
            data_keys = data_keys.union(reader.names)
        self.data_keys = list(data_keys)
        self._task_names = task_names

        with tf.variable_scope(name):
            dequeues = {k: v.dequeue() for k, v in readers.items()}
            tensors_to_enqueue = {}
            for tensor_name in data_keys:
                values = [dequeues[k][tensor_name] for k in self.reader_keys]
                tensors_to_enqueue[tensor_name] = tf.stack(values)

            # Add the dataset names so we can evaluate per-dataset
            tensors_to_enqueue['dataset'] = tf.convert_to_tensor(
                self.reader_keys)

            # Also add the dataset indexes to make it easier to split by
            # dataset
            tensors_to_enqueue[
                'dataset-index'] = tf.range(len(self.reader_keys))

            if shuffle:
                min_after_dequeue = int(0.5 * capacity)
                self._outputs = tf.train.shuffle_batch(
                    tensors_to_enqueue, batch_size, capacity=capacity,
                    min_after_dequeue=min_after_dequeue, num_threads=4, enqueue_many=True)

            else:
                self._outputs = tf.train.batch(
                    tensors_to_enqueue, batch_size, num_threads=4, capacity=capacity,
                    enqueue_many=True)

    @property
    def outputs(self):
        """The outputs of the shared examples queue."""
        return self._outputs

    @property
    def task_names(self):
        """The task names of the queue."""
        return self._task_names

    @property
    def dataset_labels(self):
        """The dataset labels that `dataset-index` corresponds to."""
        return self.reader_keys
