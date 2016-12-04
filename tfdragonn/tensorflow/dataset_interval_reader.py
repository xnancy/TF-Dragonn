from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from examples_queue import examples_queue
from interval_queue import interval_queue
from bcolz_reader_op import bcolz_interval_reader
from datasets import parse_inputs_and_intervals

"""
Primary interface for I/O for one or more datasets.
Each dataset consists of one or more datafiles, and one set of intervals and labels.
"""


def get_readers_and_tasks(processed_inputs_file, processed_intervals_file):
    """Generate a reader and examples queue for each dataset.

    Args:
        processed_inputs_file: json file with processed input data directories for each dataset
        processed_intervals_file: json file with processed intervals files for each dataset
    Returns: a tuple of:
        readers: a dictionary of an examples queue for each dataset. Each queue contains:
            intervals/start (int) - the start coordinate of the intervals
            intervals/end (int) - the end coordinate of the intervals
            intervals/chrom (string) - the chroms of the intervals
            data/{data_name} (e.g. `data/dnase_data_dir` or `data/genome_data_dir`) - the actual
                data fields that were read and possibly normalized

            iff labels are specified in the `processed_intervals_file`, also:
            labels - the labels for each interval for each task (size T)
        task names:
            a list of strings of task names, of size T
    """
    datasets = parse_inputs_and_intervals(
        processed_inputs_file, processed_intervals_file)
    task_names = datasets[list(datasets.keys())[0]]['task_names']

    examples_queues = {}
    for dataset_id, dataset in datasets.items():
        intervals = dataset['intervals']
        datafiles = dataset['inputs']
        labels = None
        if 'labels' in dataset.keys():
            labels = dataset['labels']
        examples_queues[dataset_id] = get_readers_for_dataset(
            intervals, datafiles, labels, name=dataset_id)

    return examples_queues, task_names


def get_readers_for_dataset(intervals, datafiles, labels=None, name='dataset-reader',
                            read_batch_size=128):
    """Create an input pipeline for one dataset.

    Args:
        intervals: a dict of arrays with first dimension N. Either:
            1) intervals encoded as 'chrom': (string), 'start': (int), and 'stop': (int), or
            2) intervals encoded as 'bed3': (string) TSV entries
        datafiles: a dict of genome data directory paths.
        labels: (optional) a dict of label arrays, each with first dimension N.
        name: (optional) string, name for this queue
    Returns:
        a queue reference for the examples queue
    """
    with tf.variable_scope(name):

        # Queue to store intervals to read, outputs are dequeued tensors
        interval_size = int(intervals['end'][0] - intervals['start'][0])
        to_read = interval_queue(intervals, labels, dequeue_size=read_batch_size,
                                 name='interval-queue')

        # Create a reader for each datafile
        read_values = {}
        for k, datafile in datafiles.items():
            norm_params = 'local_zscore' if k == 'dnase_data_dir' else None
            read_values[k] = bcolz_interval_reader(
                to_read, datafile, interval_size, norm_params, op_name='{}-reader'.format(k))

        # Enqueue the read values, along with labels and intervals
        interval_tensor_dict = {k: to_read[k]
                                for k in ['chrom', 'start', 'end']}
        batch_labels = None
        if 'labels' in to_read:
            batch_labels = to_read['labels']
        ex_queue = examples_queue(interval_tensor_dict, read_values, batch_labels,
                                  name='examples-queue')
        return ex_queue
