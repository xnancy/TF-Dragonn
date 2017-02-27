#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
import ntpath
import shutil

from keras import backend as K
import tensorflow as tf

import database
import genomeflow_interface
import models
import trainers
import loggers

DIR_PREFIX = '/srv/scratch/tfbinding/'
LOGDIR_PREFIX = '/srv/scratch/tfbinding/tf_logs/'

HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
VALID_CHROMS = ['chr9']

EARLYSTOPPING_KEY = 'auPRC'
EARLYSTOPPING_PATIENCE = 4
# EARLYSTOPPING_TOLERANCE = 1e-4

IN_MEMORY = False
# BATCH_SIZE = 128
BATCH_SIZE = 256
EPOCH_SIZE = 250000
# EPOCH_SIZE = 2500000 
#EPOCH_SIZE = 5000000

# TF Session Settings
DEFER_DELETE_SIZE = int(250 * 1e6)  # 250MB
GPU_MEM_PROP = 0.45  # Allows 2x sessions / gpu


logging.basicConfig(
    format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('train-wrapper')


def parse_args():
    parser = argparse.ArgumentParser('main TF-DragoNN script')
    subparsers = parser.add_subparsers(help='tf-dragonn command help', dest='command')

    train_parser = subparsers.add_parser('train', help="main training script")
    train_parser.add_argument('datasetspec', type=os.path.abspath,
                        help='Dataset parameters json file path')
    train_parser.add_argument('intervalspec', type=os.path.abspath,
                        help='Interval parameters json file path')
    train_parser.add_argument('modelspec', type=os.path.abspath,
                        help='Model parameters json file path')
    train_parser.add_argument('logdir', type=os.path.abspath,
                        help='Log directory, also used as globally unique run identifier')
    train_parser.add_argument('--visiblegpus', type=str,
                        required=True, help='Visible GPUs string')

    test_parser = subparsers.add_parser('test', help="main testing script")
    test_parser.add_argument('logdir', type=os.path.abspath,
                        help='Log directory, also used as globally unique run identifier')
    test_parser.add_argument('--visiblegpus', type=str,
                        required=True, help='Visible GPUs string')
    test_parser.add_argument('--test-size', type=int,
                             help='Limit test size, full test otherwise.')

    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def train_tf_dragonn(datasetspec, intervalspec, modelspec, logdir, visiblegpus):

    datasetspec = os.path.abspath(datasetspec)
    assert(os.path.isfile(datasetspec))
    assert(datasetspec.startswith(DIR_PREFIX))

    intervalspec = os.path.abspath(intervalspec)
    assert(os.path.isfile(intervalspec))
    assert(intervalspec.startswith(DIR_PREFIX))

    modelspec = os.path.abspath(modelspec)
    assert(os.path.isfile(modelspec))
    assert(modelspec.startswith(DIR_PREFIX))

    logdir = os.path.abspath(logdir)
    if os.path.isdir(logdir):  # remove empty directories for debugging
        if len(os.listdir(logdir)) == 0:
            shutil.rmtree(logdir)
    assert(not os.path.exists(logdir))
    assert(logdir.startswith(LOGDIR_PREFIX))
    os.makedirs(logdir)
    run_id = str(logdir.lstrip(LOGDIR_PREFIX))

    logger.info('dataspec file: {}'.format(datasetspec))
    logger.info('intervalspec file: {}'.format(intervalspec))
    logger.info('logdir path: {}'.format(logdir))
    logger.info('visiblegpus string: {}'.format(visiblegpus))

    # copy datasetspec, intervalspec, and models params to log dir
    shutil.copyfile(datasetspec, os.path.join(logdir, ntpath.basename('datasetspec.json')))
    shutil.copyfile(intervalspec, os.path.join(logdir, ntpath.basename('intervalspec.json')))
    shutil.copyfile(modelspec, os.path.join(logdir, ntpath.basename('modelspec.json')))

    # initialize logger for training
    loggers.setup_logger('trainer', os.path.join(logdir, "metrics.log"))
    trainer_logger = logging.getLogger('trainer')

    logger.info('registering with tfdragonn database')
    metadata = {}  # TODO(cprobert): save metadata here
    database.add_run(run_id, datasetspec, intervalspec,
                     modelspec, logdir, metadata)

    logger.info("Setting up keras session")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_PROP
    session = tf.Session(config=session_config)
    K.set_session(session)

    logger.info('Setting up genomeflow queues')
    data_interface = genomeflow_interface.GenomeFlowInterface(
        datasetspec, intervalspec, modelspec, VALID_CHROMS, HOLDOUT_CHROMS)
    train_queue = data_interface.get_train_queue()
    validation_queue = data_interface.get_validation_queue()

    logger.info('initializing  model and trainer')
    # jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    # with jit_scope():
    model = models.model_from_config_and_queue(modelspec, train_queue)
    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names,
                                         optimizer='adam',
                                         lr=0.0003,
                                         batch_size=BATCH_SIZE,
                                         epoch_size=EPOCH_SIZE,
                                         num_epochs=100,
                                         early_stopping_metric=EARLYSTOPPING_KEY,
                                         early_stopping_patience=EARLYSTOPPING_PATIENCE,
                                         logger=trainer_logger)
    logger.info('training model')
    trainer.train(model, train_queue, validation_queue,
                  save_best_model_to_prefix=os.path.join(logdir, "model"))


def test_tf_dragonn(logdir, visiblegpus, test_size=None):
    logdir = os.path.abspath(logdir)
    assert(os.path.exists(logdir))
    assert(logdir.startswith(LOGDIR_PREFIX))

    datasetspec = os.path.join(logdir, 'datasetspec.json')
    assert(os.path.isfile(datasetspec))

    intervalspec = os.path.join(logdir, 'intervalspec.json')
    assert(os.path.isfile(intervalspec))

    modelspec = os.path.join(logdir, 'modelspec.json')
    assert(os.path.isfile(modelspec))

    logger.info('dataspec file: {}'.format(datasetspec))
    logger.info('intervalspec file: {}'.format(intervalspec))
    logger.info('logdir path: {}'.format(logdir))
    logger.info('visiblegpus string: {}'.format(visiblegpus))

    logger.info("Setting up keras session")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_PROP
    session = tf.Session(config=session_config)
    K.set_session(session)

    logger.info('Setting up genomeflow queues')
    data_interface = genomeflow_interface.GenomeFlowInterface(
        datasetspec, intervalspec, modelspec, VALID_CHROMS, HOLDOUT_CHROMS)
    validation_queue = data_interface.get_validation_queue()

    logger.info('loading  model and trainer')
    model = models.model_from_config_and_queue(modelspec, validation_queue)
    model.load_weights(os.path.join(logdir, 'model.weights.h5'))
    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names)

    logger.info('testing model')
    classification_result = trainer.test(model, validation_queue, test_size=test_size)
    logger.info(classification_result)


def main():
    command_functions = {'train': train_tf_dragonn,
                         'test': test_tf_dragonn}
    command, args = parse_args()
    command_functions[command](**args)


if __name__ == '__main__':
    main()
