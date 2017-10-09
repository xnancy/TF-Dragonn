#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import ntpath
import shutil

from keras import backend as K
import numpy as np
import tensorflow as tf

from tfdragonn import database
from tfdragonn import models
from tfdragonn import trainers
from tfdragonn import loggers

from .genomeflow_interface import GenomeFlowInterface

# tf-binding project specific settings (only used if --is-tfbinding-project is
# specified, or the environment variable 'IS_TFBINDING_PROJECT' is set)
IS_TFBINDING_PROJECT = False
TFBINDING_DIR_PREFIX = '/srv/scratch/tfbinding/'
TFBINDING_LOGDIR_PREFIX = '/srv/scratch/tfbinding/tf_logs/'

# Default holdout and validation chromosome sets
DEFAULT_HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
DEFAULT_VALID_CHROMS = ['chr9']

# Default early stopping parameters
DEFAULT_EARLYSTOPPING_KEY = 'auPRC'
DEFAULT_EARLYSTOPPING_PATIENCE = 4

# Whether to load datasets in memory before training or read from disk
IN_MEMORY = False

# Default learning parameters
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCH_SIZE = 2500000
DEFAULT_LEARNING_RATE = 0.0003

# TF Session Settings
DEFER_DELETE_SIZE = int(250 * 1e6)  # 250MB
GPU_MEM_PROP = 0.45  # Allows 2x sessions / gpu

backend = K.backend()
if backend != 'tensorflow':
    raise ValueError(
        'Only the keras tensorflow backend is supported, currently using {}'.format(backend))


class BaseModelRunner(object):
    command = None

    def __init__(self):
        self._logger_name = 'tfdragonn-{}'.format(self.command)
        self._logger = loggers.get_logger(self._logger_name)

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser('tfdragonn {}'.format(cls.command))
        parser.add_argument('datasetspec', type=os.path.abspath,
                            help='Dataset parameters json file path')
        parser.add_argument('intervalspec', type=os.path.abspath,
                            help='Interval parameters json file path')
        parser.add_argument('modelspec', type=os.path.abspath,
                            help='Model parameters json file path')
        parser.add_argument('logdir', type=os.path.abspath,
                            help='Log directory, also used as globally unique run identifier')
        parser.add_argument('--visiblegpus', type=str,
                            required=True, help='Visible GPUs string')
        parser.add_argument('--maxexs', type=int,
                            help='max number of examples', default=None)
        parser.add_argument('--is-tfbinding-project', action='store_true',
                            help='Use tf-binding project specific settings')
        cls.add_additional_args(parser)
        return parser

    @classmethod
    def add_additional_args(cls, parser):
        """Add any class-specific arguments to the parser."""
        pass

    @classmethod
    def parse_args(cls, args):
        parser = cls.get_parser()
        args = parser.parse_args(args)
        return args

    def run_from_args(self, command, args):
        if command != self.command:
            raise ValueError('Wrong command "{}" for runner "{}". Expecting "{}".'.format(
                command, self.__class__.__name__, self.command))
        args = self.parse_args(args)
        self.start_run(args)

    def start_run(self, params):
        """Main entrypoiny for running a model."""
        if params.is_tfbinding_project or 'IS_TFBINDING_PROJECT' in os.environ:
            global IS_TFBINDING_PROJECT
            IS_TFBINDING_PROJECT = True
            run_id = str(params.logdir.lstrip(TFBINDING_LOGDIR_PREFIX))
            if self.command == 'train':
                database.add_run(run_id, params.datasetspec, params.intervalspec,
                                 params.modelspec, params.logdir)
        self.validate_paths(params)
        if self.command == 'train':
            os.makedirs(params.logdir)
        loggers.add_logdir(self._logger_name, params.logdir)
        self.setup_keras_session(params.visiblegpus)
        self.run(params)

    def run(self, params):
        raise NotImplementedError('Model runners must implement run')

    @staticmethod
    def setup_keras_session(visiblegpus):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
        session_config = tf.ConfigProto()
        session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
        session_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_PROP
        session = tf.Session(config=session_config)
        K.set_session(session)

    @classmethod
    def validate_paths(cls, params):
        for specfile in [params.datasetspec, params.intervalspec, params.modelspec]:
            cls.validate_specfile(specfile)
        # remove empty directories for debugging
        if os.path.isdir(params.logdir):
            if len(os.listdir(params.logdir)) == 0:
                shutil.rmtree(params.logdir)
        assert(not os.path.exists(params.logdir))
        if IS_TFBINDING_PROJECT:
            assert(params.logdir.startswith(TFBINDING_LOGDIR_PREFIX))

    @staticmethod
    def validate_specfile(specfile):
        if not os.path.isfile(specfile):
            raise FileNotFoundError(
                'Specfile {} does not exist'.format(specfile))
        if IS_TFBINDING_PROJECT:
            assert(specfile.startswith(TFBINDING_DIR_PREFIX))


class TrainRunner(BaseModelRunner):
    command = 'train'

    @classmethod
    def add_additional_args(cls, parser):
        parser.add_argument('--validation-intervalspec',
                            type=os.path.abspath,
                            help='Heldout celltype intervalspec to use as validation set, default: None',
                            default=None)
        parser.add_argument('--holdout-chroms',
                            type=json.loads,
                            help='Set of chroms to holdout entirely from training/validation as a json string, default: "{}"'.format(
                                str(DEFAULT_HOLDOUT_CHROMS)),
                            default=DEFAULT_HOLDOUT_CHROMS)
        parser.add_argument('--valid-chroms',
                            type=json.loads,
                            help='Set of chroms to holdout from training and use for validation as a json string, default: "{}"'.format(
                                str(DEFAULT_VALID_CHROMS)),
                            default=DEFAULT_VALID_CHROMS)
        parser.add_argument('--learning-rate',
                            type=float,
                            help='Learning rate (float), default: {}'.format(DEFAULT_LEARNING_RATE),
                            default=DEFAULT_LEARNING_RATE)
        parser.add_argument('--batch-size',
                            type=int,
                            help='Batch size (int), default: {}'.format(DEFAULT_BATCH_SIZE),
                            default=DEFAULT_BATCH_SIZE)
        parser.add_argument('--epoch-size',
                            type=int,
                            help='Epoch size (int), default: {}'.format(DEFAULT_EPOCH_SIZE),
                            default=DEFAULT_EPOCH_SIZE)
        parser.add_argument('--early-stopping-metric',
                            type=str,
                            help='Early stopping metric key, default: {}'.format(
                                DEFAULT_EARLYSTOPPING_KEY),
                            default=DEFAULT_EARLYSTOPPING_KEY)
        parser.add_argument('--early-stopping-patience',
                            type=int,
                            help='Early stopping patience (int), default: {}'.format(
                                DEFAULT_EARLYSTOPPING_PATIENCE),
                            default=DEFAULT_EARLYSTOPPING_PATIENCE)

    def run(self, params):
        shutil.copyfile(params.datasetspec, os.path.join(
            params.logdir, ntpath.basename('datasetspec.json')))
        shutil.copyfile(params.intervalspec, os.path.join(
            params.logdir, ntpath.basename('intervalspec.json')))
        shutil.copyfile(params.modelspec, os.path.join(
            params.logdir, ntpath.basename('modelspec.json')))

        data_interface = GenomeFlowInterface(
            params.datasetspec, params.intervalspec, params.modelspec, params.logdir,
            validation_chroms=params.valid_chroms,
            holdout_chroms=params.holdout_chroms,
            validation_intervalspec=params.validation_intervalspec,
            logger=self._logger)
        train_queue = data_interface.get_train_queue()
        validation_queue = data_interface.get_validation_queue()

        trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names,
                                             optimizer='adam',
                                             lr=params.learning_rate,
                                             batch_size=params.batch_size,
                                             epoch_size=params.epoch_size,
                                             num_epochs=100,
                                             early_stopping_metric=params.early_stopping_metric,
                                             early_stopping_patience=params.early_stopping_patience,
                                             logger=self._logger)

        model = models.model_from_minimal_config(
            params.modelspec, train_queue.output_shapes, len(data_interface.task_names))

        trainer.train(model, train_queue, validation_queue,
                      save_best_model_to_prefix=os.path.join(params.logdir, "model"))



class TestRunner(BaseModelRunner):
    command = 'test'

    def run(self, params):
        data_interface = GenomeFlowInterface(
            params.datasetspec, params.intervalspec, params.modelspec, params.logdir)
        validation_queue = data_interface.get_validation_queue()
        model = models.model_from_minimal_config(
            params.modelspec, validation_queue.output_shapes, len(data_interface.task_names))
        model.load_weights(os.path.join(
            params.logdir, 'model.weights.h5'))
        trainer = trainers.ClassifierTrainer(
            task_names=data_interface.task_names)
        classification_result = trainer.test(model, validation_queue, test_size=params.maxexs)
        self._logger.info('\n{}'.format(classification_result))

    @classmethod
    def validate_paths(cls, params):
        for specfile in [params.datasetspec, params.intervalspec, params.modelspec]:
            cls.validate_specfile(specfile)
        assert(os.path.exists(params.logdir))
        if IS_TFBINDING_PROJECT:
            assert(params.logdir.startswith(TFBINDING_LOGDIR_PREFIX))


class PredictRunner(TestRunner):
    command = 'predict'

    @classmethod
    def add_additional_args(cls, parser):
        parser.add_argument('prefix',
                            type=str,
                            help='Prefix for files with predictions')
        parser.add_argument('--flank-size',
                            type=int,
                            help='Size of flank in input intervals, flanks are trimmed before writing intervals to file. default: 400',
                            default=400)

    def run(self, params):
        data_interface = GenomeFlowInterface(
            params.datasetspec, params.intervalspec, params.modelspec, params.logdir, shuffle=False, pos_sampling_rate=None)
        example_queues = {dataset_id: data_interface.get_example_queue(dataset_values, dataset_id,
                                                                       num_epochs=1,
                                                                       input_names=data_interface.input_names,
                                                                       enqueues_per_thread=[128, 1])
                          for dataset_id, dataset_values in data_interface.dataset.items()}
        model = models.model_from_minimal_config(
            params.modelspec, example_queues.values()[0].output_shapes, len(data_interface.task_names))
        model.load_weights(os.path.join(
            params.logdir, 'model.weights.h5'))
        trainer = trainers.ClassifierTrainer(
            task_names=data_interface.task_names)

        for dataset_id, example_queue in example_queues.items():
            self._logger.info('generating predictions for dataset {}'.format(dataset_id))
            intervals, predictions = trainer.predict(model, example_queue)

            # trim flanks
            intervals['start'] += params.flank_size
            intervals['end'] -= params.flank_size

            # write intervals and per task predictions to file
            for task_indx, task_name in enumerate(data_interface.task_names):
                prediction_data = np.column_stack((intervals['chrom'],
                                                   intervals['start'],
                                                   intervals['end'],
                                                   predictions[:, task_indx]))
                prediction_fname = "{}.{}.{}.tab.gz".format(params.prefix, task_name, dataset_id)
                np.savetxt(prediction_fname, prediction_data, delimiter='\t', fmt='%s')
                self._logger.info("\nSaved {} predictions in dataset {} to {}".format(
                    task_name, dataset_id, prediction_fname))
            self._logger.info('Done!')
