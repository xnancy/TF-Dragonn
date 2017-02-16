from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import genomeflow as gf

import datasets

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
                 validation_chroms=[], holdout_chroms=[]):
        self.datasetspec = datasetspec
        self.intervalspec = intervalspec
        self.modelspec = modelspec

        self.training_dataset, self.validation_dataset = \
            datasets.parse_inputs_and_intervals_with_holdout(
                self.datasetspec, self.intervalspec,
                validation_chroms, holdout_chroms)
        self.task_names = self.training_dataset.values()[0]['task_names']

        self.num_train_exs = sum(
            self.training_dataset[k]['intervals']['chrom'].shape[0]
            for k in self.training_dataset.keys())
        self.num_validation_exs = sum(
            self.validation_dataset[k]['intervals']['chrom'].shape[0]
            for k in self.validation_dataset.keys())

    def get_train_queue(self):
        return self.get_queue(self.training_dataset)

    def get_validation_queue(self, num_epochs=1, asynchronous_enqueues=False):
        return self.get_queue(
            self.validation_dataset, num_epochs, asynchronous_enqueues)

    def get_queue(self, dataset, num_epochs=None, asynchronous_enqueues=True):
        examples_queues = {
            dataset_id: self.get_example_queue(dataset_values, dataset_id, num_epochs)
            for dataset_id, dataset_values in dataset.items()
        }
        shared_examples_queue = self.get_shared_examples_queue(
            examples_queues, asynchronous_enqueues=asynchronous_enqueues)
        return shared_examples_queue

    def get_example_queue(self, dataset, dataset_id, num_epochs=None):
        intervals = dataset['intervals']
        inputs = dataset['inputs']
        labels = dataset['labels']

        interval_queue = gf.io.IntervalQueue(
            intervals, labels, name='{}-interval-queue'.format(dataset_id),
            num_epochs=num_epochs, capacity=10000, shuffle=False, summary=True)

        data_sources = {k: self.get_data_source(k, v) for k, v in inputs.items()}

        examples_queue = gf.io.ExampleQueue(
            interval_queue, data_sources, num_threads=1,
            enqueue_batch_size=128, capacity=2048,
            name='{}-example-queue'.format(dataset_id))

        return examples_queue

    def get_shared_examples_queue(self, examples_queues, asynchronous_enqueues=True):
        shared_examples_queue = gf.io.MultiDatasetExampleQueue(
            examples_queues, num_threads=1, enqueue_batch_size=128,
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
        if extractor_type == 'bed': # parse data specs
            data_path = data_specs['filepath']
            options = data_type2options[data_type].copy()
            options.update(data_specs['options'])
        return gf.io.DataSource(data_path, extractor_type, options)
