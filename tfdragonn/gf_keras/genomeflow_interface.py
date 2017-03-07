from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import utils

import genomeflow as gf

import datasets
import models

data_type2extractor = {
    'genome_data_dir': 'bcolz_array',
    'dnase_data_dir': 'bcolz_array',
    'tss_counts': 'bed',
    'dhs_counts': 'bed',
    'tss_mean_tpm': 'bed',
    'tss_max_tpm': 'bed'
}
data_type2options = {
    'tss_counts': {
        'op': 'count'
    },
    'dhs_counts': {
        'op': 'count'
    },
    'tss_mean_tpm': {
        'op': 'mean',
        'norm_params': 'asinh_zscore'
    },
    'tss_max_tpm': {
        'op': 'max',
        'norm_params': 'asinh_zscore'
    }
}


class GenomeFlowInterface(object):

    def __init__(self, datasetspec, intervalspec, modelspec,
                 validation_chroms=[], holdout_chroms=[],
                 shuffle=True, pos_sampling_rate=0.05):
        self.datasetspec = datasetspec
        self.intervalspec = intervalspec
        self.modelspec = modelspec
        input_names = models.model_inputs_from_config(modelspec)
        self.input_names = [input_name.split('/')[1]
                            for input_name in input_names]
        self.shuffle = shuffle

        self.training_dataset, self.validation_dataset = \
            datasets.parse_inputs_and_intervals_with_holdout(
                self.datasetspec, self.intervalspec,
                validation_chroms, holdout_chroms)
        self.task_names = self.training_dataset.values()[0]['task_names']
        if pos_sampling_rate is not None and len(self.task_names) > 1:
            raise ValueError("pos_sampling_rate is not supported for >1 tasks")

        self.pos_sampling_rate = pos_sampling_rate
        self.num_train_exs = sum(
            self.training_dataset[k]['intervals']['chrom'].shape[0]
            for k in self.training_dataset.keys())
        self.num_validation_exs = sum(
            self.validation_dataset[k]['intervals']['chrom'].shape[0]
            for k in self.validation_dataset.keys())

    def get_train_queue(self):
        return self.get_queue(self.training_dataset,
                              pos_sampling_rate=self.pos_sampling_rate,
                              input_names=self.input_names,
                              shuffle=self.shuffle)

    def get_validation_queue(self, num_epochs=1, asynchronous_enqueues=False,
                             enqueues_per_thread=[128, 1]):
        return self.get_queue(
            self.validation_dataset, num_epochs, asynchronous_enqueues,
            input_names=self.input_names, enqueues_per_thread=enqueues_per_thread)

    def get_queue(self, dataset, num_epochs=None, asynchronous_enqueues=True,
                  pos_sampling_rate=None, input_names=None, shuffle=False,
                  enqueues_per_thread=[128]):
        examples_queues = {
            dataset_id: self.get_example_queue(dataset_values, dataset_id,
                                               num_epochs=num_epochs,
                                               pos_sampling_rate=pos_sampling_rate,
                                               input_names=input_names,
                                               shuffle=shuffle,
                                               enqueues_per_thread=enqueues_per_thread)
            for dataset_id, dataset_values in dataset.items()
        }
        shared_examples_queue = self.get_shared_examples_queue(
            examples_queues, asynchronous_enqueues=asynchronous_enqueues,
            enqueues_per_thread=enqueues_per_thread)
        return shared_examples_queue

    def get_example_queue(self, dataset, dataset_id,
                          num_epochs=None, pos_sampling_rate=None,
                          input_names=None, shuffle=False, enqueues_per_thread=[128]):
        intervals = dataset['intervals']
        inputs = dataset['inputs']
        if 'labels' in dataset:
            labels = dataset['labels']
        else:
            labels = None
        if shuffle: # shuffle intervals and labels
            chroms = intervals['chrom']
            starts = intervals['start']
            ends = intervals['end']
            chroms, starts, ends, labels = utils.shuffle(
                chroms, starts, ends, labels, random_state=0)
            intervals = {'chrom': chroms, 'start': starts, 'end': ends}
        if pos_sampling_rate is not None:
            # construct separate interval queues for positive and negative
            # intervals
            pos_indxs = labels == 1
            neg_indxs = labels == 0

            neg_indxs[20000000:] = False # temporary hack to overcome 2Gb limit

            # assumes single task labels!
            pos_labels = labels[pos_indxs][:, None]
            neg_labels = labels[neg_indxs][:, None]

            pos_indxs = pos_indxs.squeeze()  # need 1d indices for interval arrays
            neg_indxs = neg_indxs.squeeze()

            pos_intervals = {k: v[pos_indxs] for k, v in intervals.items()}
            neg_intervals = {k: v[neg_indxs] for k, v in intervals.items()}

            pos_interval_queue = gf.io.IntervalQueue(
                pos_intervals, pos_labels, name='{}-pos-interval-queue'.format(dataset_id),
                num_epochs=num_epochs, capacity=50000, min_after_dequeue=40000,
                shuffle=shuffle, summary=True)
            neg_interval_queue = gf.io.IntervalQueue(
                neg_intervals, neg_labels, name='{}-neg-interval-queue'.format(dataset_id),
                num_epochs=num_epochs, capacity=50000, min_after_dequeue=40000,
                shuffle=shuffle, summary=True)

            # sample from both queues using a shared intervals queue
            interval_queue_ratios = {pos_interval_queue: pos_sampling_rate,
                                     neg_interval_queue: 1 - pos_sampling_rate}
            interval_queue = gf.io.SharedIntervalQueue(
                interval_queue_ratios, name='{}-shared-interval-queue'.format(
                    dataset_id), capacity=50000, enqueue_batch_size=128)
        else:
            if intervals['chrom'].shape[0] > 20000000: # temporary hack to overcome 2Gb limit
                labels = labels[:20000000]
                intervals = {k: v[:20000000] for k, v in intervals.items()}

            interval_queue = gf.io.IntervalQueue(
                intervals, labels, name='{}-interval-queue'.format(dataset_id),
                num_epochs=num_epochs, capacity=50000, shuffle=shuffle,
                min_after_dequeue=40000, summary=True)

        if input_names is not None:  # use only these inputs in the example queue
            assert all([input_name in inputs.keys()
                        for input_name in input_names])
            data_sources = {k: self.get_data_source(k, v) for k, v in inputs.items()
                            if k in input_names}
        else:
            data_sources = {k: self.get_data_source(
                k, v) for k, v in inputs.items()}

        examples_queue = gf.io.ExampleQueue(
            interval_queue, data_sources, enqueues_per_thread=enqueues_per_thread,
            capacity=2048, name='{}-example-queue'.format(dataset_id))

        return examples_queue

    def get_shared_examples_queue(self, examples_queues, asynchronous_enqueues=True,
                                  enqueues_per_thread=[128]):
        shared_examples_queue = gf.io.MultiDatasetExampleQueue(
            examples_queues, enqueues_per_thread=enqueues_per_thread,
            capacity=2048, name='multi-dataset-example-queue',
            asynchronous_enqueues=asynchronous_enqueues)
        return shared_examples_queue

    def get_data_source(self, data_type, data_specs):
        """
        data_specs is either the file path for bcolz data
        or dictionary with specs for bed data.
        """
        extractor_type = data_type2extractor[data_type]
        options = {}
        data_path = data_specs
        if extractor_type == 'bed':  # parse data specs
            data_path = data_specs['filepath']
            options = data_type2options[data_type].copy()
            options.update(data_specs['options'])
        return gf.io.DataSource(data_path, extractor_type, options)
