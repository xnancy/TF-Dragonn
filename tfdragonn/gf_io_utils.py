import tensorflow as tf
from tensorflow.python.training import coordinator


class ExampleQueueIterator(object):

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_examples(self):
        return self._queue.num_examples

    def __init__(self, queue, num_exs_batch=128, num_epochs=None, num_exs_epoch=None,
                 allow_smaller_final_batch=False):
        """
        If num_epochs is set, limit number of epochs to iterate over.

        If num_exs_epoch is None, use queue.num_examples as num_exs_epoch. Else,
            use num_exs_epoch as the epoch size.
        """
        if allow_smaller_final_batch:
            queue_outputs = queue.dequeue_up_to(num_exs_batch)
        else:
            queue_outputs = queue.dequeue_many(num_exs_batch)

        # Run queue on the CPU only (use 0 GPUs)
        config = tf.ConfigProto(device_count={'GPU': 0})
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("*****START QUEUE RUNNERS*****")
        self._coord = coordinator.Coordinator()
        self._queue_runner_threads = tf.train.start_queue_runners(session, self._coord)

        self._batch_size = num_exs_batch
        self._session = session
        self._queue = queue
        self._queue_outputs = queue_outputs

        if num_exs_epoch is None:
            num_exs_epoch = self._queue.num_examples
        self._epoch_size = num_exs_epoch

        if num_epochs is None:
            self._len = None
        else:
            self._len = num_epochs * self._epoch_size

        self._num_examples_left = self._len

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def next(self):
        if self._len is not None:
            if self._num_examples_left <= 0:
                self.close()
                raise StopIteration

        batch = self._session.run(self._queue_outputs)

        self._num_examples_left -= batch.values()[0].shape[0]
        return batch

    def close(self):
        coord = getattr(self, "_coord", None)
        if coord is not None:
            self._coord.request_stop()
            self._coord.join(self._queue_runner_threads)

    def __next__(self):
        return self.next()

    def __del__(self):
        sess = getattr(self, "_session", None)
        if sess is not None:
            sess.close()
        del sess
