import tensorflow as tf

class ExampleQueueIterator(object):

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_examples(self):
        return self._queue.num_examples

    def __init__(self, queue, batch_size=128, num_epochs=None):
        queue_outputs = queue.dequeue_many(batch_size)

        # Run queue on the CPU only (use 0 GPUs)
        config = tf.ConfigProto(device_count={'GPU': 0})
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print "*****START QUEUE RUNNERS*****"
        queue_runner_threads = tf.train.start_queue_runners(session)

        self._batch_size = batch_size
        self._session = session
        self._queue = queue
        self._queue_outputs = queue_outputs
        if num_epochs is not None:
            self._len = num_epochs * self._queue.num_examples
        else:
            self._len = None
        self._num_examples_left = self._len

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def next(self):
        if self._len is not None:
            if self._num_examples_left <= 0:
                raise StopIteration

            self._num_examples_left -= self._batch_size
        try:
            batch = self._session.run(self._queue_outputs)
        except tf.errors.OutOfRangeError:
            batch = self._session.run(self._queue_outputs)

        return batch

    def __next__(self):
        return self.next()

    def __del__(self):
        if self._session:
            self._session.close()
