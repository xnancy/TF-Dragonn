import numpy as np
import tensorflow as tf

from genomeflow.io import IntervalQueue
from genomeflow.io import ExampleQueue

class ExampleQueueIterator(object):

    @property
    def batch_size(self):
        return self._batch_size

    def __init__(self, queue, batch_size=256,
                 num_epochs=None,
                 num_threads=1, shuffle=False, random_seed=228,
                 in_memory=False):
        """
        gdl_file = GDLFile(source_gdl, base_dir=base_dir)

        intervals = dict(zip(
            ('chrom', 'start', 'end'),
            map(np.asarray,
                zip(*[(interval.chrom, interval.start, interval.stop)
                      for interval in gdl_file.intervals]))))
        datafile_paths = {
            data_name: feo_tuple[0]
            for data_name, feo_tuple in
            zip(data_names, gdl_file.file_extractor_opts)
        }

        datafile_paths = {'atac': base_dir}

        interval_queue_capacity = 10000
        interval_queue = IntervalQueue(
            intervals, labels=gdl_file.labels, num_epochs=num_epochs,
            capacity=interval_queue_capacity, shuffle=shuffle,
            min_after_dequeue=0.9 * interval_queue_capacity)

        queue = ExampleQueue(
            interval_queue, datafile_paths,
            enqueue_batch_size=256, capacity=4096, num_threads=num_threads,
            in_memory=in_memory)
        """
        queue_outputs = queue.dequeue_many(batch_size)

        # Run these queues on the CPU only (use 0 GPUs)
        config = tf.ConfigProto(device_count={'GPU': 0})
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        # We don't really need the return value actually

        print "*****START QUEUE RUNNERS*****"
        queue_runner_threads = tf.train.start_queue_runners(session)

        #self._label_names = label_names
        self._batch_size = batch_size
        #self._gdl_file = gdl_file
        self._session = session
        self._queue = queue
        self._queue_outputs = queue_outputs
        if num_epochs is not None:
            self._len = num_epochs * self._queue._num_examples
        else:
            self._len = None
        self._num_examples_left = self._len
        #self._data_outputs = map('data/{}'.format, data_names)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def next(self):
        if self._len is not None:
            if self._num_examples_left <= 0:
                raise StopIteration

            self._num_examples_left -= self._batch_size
        batch = self._session.run(self._queue_outputs)
        return batch
        #examples = {output: expand_dims(batch[output])
        #            for output in self._data_outputs}
        #labels = {output: col.T
        #          for output, col in zip(self._label_names, batch['labels'].T)}
        #return examples, labels

    def __next__(self):
        return self.next()

    def __del__(self):
        if self._session:
            self._session.close()
