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
import tensorflow as tf

import database
from genomeflow_interface import GenomeFlowInterface
import models
import trainers
import loggers


# tfbinding project specific settings (only used if --tfbinding-project is
# specified)
IS_TFBINDING_PROJECT = False
TFBINDING_DIR_PREFIX = '/srv/scratch/tfbinding/'
TFBINDING_LOGDIR_PREFIX = '/srv/scratch/tfbinding/tf_logs/'

DEFAULT_HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
DEFAULT_VALID_CHROMS = ['chr9']

DEFAULT_EARLYSTOPPING_KEY = 'auPRC'
DEFAULT_EARLYSTOPPING_PATIENCE = 4

IN_MEMORY = False
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
        args = self.parse_args(args)
        self.start_run(command, args)

    def start_run(self, command, params):
        """Main entrypoiny for running a model."""
        loggers.add_logdir(self._logger_name, params.logdir)
        self.setup_keras_session(params.visiblegpus)
        self.run(command, params)

    def run(self, command, params):
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
        parser.add_argument('--holdout-chroms',
                            type=json.loads,
                            help='Test chroms to holdout from training/validation',
                            default=DEFAULT_HOLDOUT_CHROMS)
        parser.add_argument('--valid-chroms',
                            type=json.loads,
                            help='Validation to holdout from training and use for validation',
                            default=DEFAULT_HOLDOUT_CHROMS)
        parser.add_argument('--learning-rate',
                            type=float,
                            help='Learning rate (float)',
                            default=DEFAULT_LEARNING_RATE)
        parser.add_argument('--batch-size',
                            type=int,
                            help='Batch size (int)',
                            default=DEFAULT_BATCH_SIZE)
        parser.add_argument('--epoch-size',
                            type=int,
                            help='Epoch size (int)',
                            default=DEFAULT_EPOCH_SIZE)
        parser.add_argument('--early-stopping-metric',
                            type=str,
                            help='Early stopping metric key',
                            default=DEFAULT_EARLYSTOPPING_KEY)
        parser.add_argument('--early-stopping-patience',
                            type=int,
                            help='Early stopping patience (int)',
                            default=DEFAULT_EARLYSTOPPING_PATIENCE)

    def run(self, command, params):
        data_interface = GenomeFlowInterface(
            params.datasetspec, params.intervalspec, params.modelspec,
            validation_chroms=params.valid_chroms,
            holdout_chroms=params.holdout_chroms)
        train_queue = data_interface.get_train_queue()
        validation_queue = data_interface.get_validation_queue()

        trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names,
                                             optimizer='adam',
                                             lr=params.learning_rate,
                                             batch_size=params.batch_size,
                                             epoch_size=params.epoch_size,
                                             num_epochs=100,
                                             early_stopping_metric=params.early_stopping_metric,
                                             early_stopping_patience=params.early_stopping_patience)

        model = models.model_from_minimal_config(
            params.modelspec, train_queue.output_shapes, len(data_interface.task_names))

        trainer.train(model, train_queue, validation_queue,
                      save_best_model_to_prefix=os.path.join(params.logdir, "model"))

        shutil.copyfile(params.datasetspec, os.path.join(
            params.logdir, ntpath.basename('datasetspec.json')))
        shutil.copyfile(params.intervalspec, os.path.join(
            params.logdir, ntpath.basename('intervalspec.json')))
        shutil.copyfile(params.modelspec, os.path.join(
            params.logdir, ntpath.basename('modelspec.json')))


class TestRunner(BaseModelRunner):
    command = 'test'

    def run(self, command, params):
        data_interface = GenomeFlowInterface(
            params.datasetspec, params.intervalspec, params.modelspec)
        validation_queue = data_interface.get_validation_queue()
        model = models.model_from_config_and_queue(
            params.modelspec, validation_queue)
        model.load_weights(os.path.join(
            params.logdir, 'model.weights.h5'))
        trainer = trainers.ClassifierTrainer(
            task_names=data_interface.task_names)
        trainer.test(model, validation_queue, test_size=params.numexs)


class PredictRunner(BaseModelRunner):
    command = 'predict'

    def run(self, command, params):
        data_interface = GenomeFlowInterface(
            params.datasetspec, params.intervalspec, params.modelspec)
        validation_queue = data_interface.get_validation_queue()
        model = models.model_from_config_and_queue(
            params.modelspec, validation_queue)
        model.load_weights(os.path.join(
            params.logdir, 'model.weights.h5'))
        trainer = trainers.ClassifierTrainer(
            task_names=data_interface.task_names)
        trainer.test(model, validation_queue, test_size=params.numexs)
