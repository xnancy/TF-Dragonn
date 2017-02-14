from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import genomeflow as gf

import datasets

Datatype2Extractor = {
    'genome_data_dir': 'bcolz',
    'dnase_data_dir': 'bcolz',
    'gencode_tss': 'bed',
    'gencode_annotation': 'bed',
    'gencode_polyA': 'bed',
    'gencode_lncRNA': 'bed',
    'gencode_tRNA': 'bed',
    'expression_tsv': 'bed',
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

        self.num_train_exs = sum(
            self.training_dataset[k]['intervals'].shape[0]
            for k in self.training_dataset.keys())
        self.num_validation_exs = sum(
            self.validation_dataset[k]['intervals'].shape[0]
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
        dataset_fields = dataset[dataset_id]

        intervals = dataset_fields['intervals']
        inputs = dataset_fields['inputs']
        labels = dataset_fields['labels']
        self.task_names = dataset_fields['task_names']

        interval_queue = gf.io.IntervalQueue(
            intervals, labels, name='{}-interval-queue'.format(dataset_id),
            num_epochs=num_epochs, capacity=10000, shuffle=False, summary=True)

        data_sources = {k: self.get_data_source(v) for k, v in inputs.items()}

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

    def get_data_source(self, data_type, data_path):
        # TODO: this should check self.datasetspec or self.modelspec for
        # extraction params, like `op` and `window_half_widths`
        extractor_type = Datatype2Extractor[data_type]
        options = {}
        if extractor_type == 'bed':
            options = {'op': 'max', 'window_half_widths': [1000, 10000]}
        return gf.io.DataSource(data_path, extractor_type, options)
