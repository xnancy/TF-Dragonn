#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import ntpath
import shutil

from keras import backend as K
import tensorflow as tf

import database
from genomeflow_interface import GenomeFlowInterface
import models
import trainers
import loggers
from model_runner_params import get_model_run_params


# tfbinding project specific settings (only used if --tfbinding-project is
# specified)
IS_TFBINDING_PROJECT = False
TFBINDING_DIR_PREFIX = '/srv/scratch/tfbinding/'
TFBINDING_LOGDIR_PREFIX = '/srv/scratch/tfbinding/tf_logs/'

TFBINDING_HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
TFBINDING_VALID_CHROMS = ['chr9']

TFBINDING_EARLYSTOPPING_KEY = 'auPRC'
TFBINDING_EARLYSTOPPING_PATIENCE = 4

IN_MEMORY = False
BATCH_SIZE = 256
EPOCH_SIZE = 2500000
LEARNING_RATE = 0.0003

# TF Session Settings
DEFER_DELETE_SIZE = int(250 * 1e6)  # 250MB
GPU_MEM_PROP = 0.45  # Allows 2x sessions / gpu

LOGGER_NAME = 'tfdragonn'
_logger = loggers.get_logger(LOGGER_NAME)

backend = K.backend()
if backend != 'tensorflow':
    raise ValueError(
        'Only the keras tensorflow backend is supported, currently using {}'.format(backend))


def parse_args(args):
    parser = argparse.ArgumentParser('TF-DragoNN model runner')
    parser.add_argument('command', type=str,
                        help='command: train, test, predict')
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
    parser.add_argument('--max_examples', type=int,
                        help='max number of examples', default=None)
    parser.add_argument('--tfbinding-project', action='store_true', help='Use tfbinding '
                        'project presets (logging, path checks, database, etc)')

    args = parser.parse_args(args)
    return args


def run_from_args(command, args):
    args = parse_args(args)
    run(command, args.datasetspec, args.intervalspec,
        args.modelspec, args.logdir, args.visiblegpus, args.max_examples)


def run(command, datasetspec, intervalspec, modelspec, logdir, visiblegpus, max_examples=None):
    command_functions = {
        'train': train,
        'test': test,
    }
    loggers.add_logdir(LOGGER_NAME, logdir)
    model_run_params = get_model_run_params(
        datasetspec, intervalspec, modelspec, logdir, visiblegpus, numexs=max_examples)
    run_fn = command_functions[command]
    run_fn(model_run_params, visiblegpus)


def run_model(runner, model_run_params, visiblegpus):
    """Base method for running a model (train, test, predict)"""
    validate_paths(model_run_params)

    _logger.info('model_run_params.datasetspec file: {}'.format(
        model_run_params.datasetspec))
    _logger.info('model_run_params.intervalspec file: {}'.format(
        model_run_params.intervalspec))
    _logger.info('model_run_params.logdir path: {}'.format(
        model_run_params.logdir))
    _logger.info('visiblegpus string: {}'.format(visiblegpus))

    setup_keras_session(visiblegpus)
    runner(model_run_params.datasetspec, model_run_params.intervalspec,
           model_run_params.modelspec, model_run_params.logdir)


def train(model_run_params):
    data_interface = GenomeFlowInterface(
        model_run_params.datasetspec, model_run_params.intervalspec, model_run_params.modelspec,
        validation_chroms=TFBINDING_VALID_CHROMS,
        holdout_chroms=TFBINDING_HOLDOUT_CHROMS)
    train_queue = data_interface.get_train_queue()
    validation_queue = data_interface.get_validation_queue()

    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names,
                                         optimizer='adam',
                                         lr=LEARNING_RATE,
                                         batch_size=BATCH_SIZE,
                                         epoch_size=EPOCH_SIZE,
                                         num_epochs=100,
                                         early_stopping_metric=TFBINDING_EARLYSTOPPING_KEY,
                                         early_stopping_patience=TFBINDING_EARLYSTOPPING_PATIENCE)

    model = models.model_from_minimal_config(
        model_run_params.modelspec, train_queue.output_shapes, len(data_interface.task_names))

    trainer.train(model, train_queue, validation_queue,
                  save_best_model_to_prefix=os.path.join(model_run_params.logdir, "model"))

    shutil.copyfile(model_run_params.datasetspec, os.path.join(
        model_run_params.logdir, ntpath.basename('model_run_params.datasetspec.json')))
    shutil.copyfile(model_run_params.intervalspec, os.path.join(
        model_run_params.logdir, ntpath.basename('model_run_params.intervalspec.json')))
    shutil.copyfile(model_run_params.modelspec, os.path.join(
        model_run_params.logdir, ntpath.basename('model_run_params.modelspec.json')))


def test(model_run_params):
    data_interface = GenomeFlowInterface(
        model_run_params.datasetspec, model_run_params.intervalspec, model_run_params.modelspec)
    validation_queue = data_interface.get_validation_queue()
    model = models.model_from_config_and_queue(
        model_run_params.modelspec, validation_queue)
    model.load_weights(os.path.join(
        model_run_params.logdir, 'model.weights.h5'))
    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names)
    trainer.test(model, validation_queue, test_size=model_run_params.numexs)


def predict(model_run_params):
    data_interface = GenomeFlowInterface(
        model_run_params.datasetspec, model_run_params.intervalspec, model_run_params.modelspec)
    validation_queue = data_interface.get_validation_queue()
    model = models.model_from_config_and_queue(
        model_run_params.modelspec, validation_queue)
    model.load_weights(os.path.join(
        model_run_params.logdir, 'model.weights.h5'))
    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names)
    trainer.test(model, validation_queue, test_size=model_run_params.numexs)


def validate_paths(model_run_params):
    for specfile in [model_run_params.datasetspec, model_run_params.intervalspec, model_run_params.modelspec]:
        validate_specfile(specfile)
    # remove empty directories for debugging
    if os.path.isdir(model_run_params.logdir):
        if len(os.listdir(model_run_params.logdir)) == 0:
            shutil.rmtree(model_run_params.logdir)
    assert(not os.path.exists(model_run_params.logdir))
    if IS_TFBINDING_PROJECT:
        assert(model_run_params.logdir.startswith(TFBINDING_LOGDIR_PREFIX))


def validate_specfile(specfile):
    if not os.path.isfile(specfile):
        raise FileNotFoundError('Specfile {} does not exist'.format(specfile))
    if IS_TFBINDING_PROJECT:
        assert(specfile.startswith(TFBINDING_DIR_PREFIX))


def setup_keras_session(visiblegpus):
    _logger.info("Setting up keras session")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_PROP
    session = tf.Session(config=session_config)
    K.set_session(session)
