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

"""
TRAIN_DIRNAME = 'train'
VALID_DIRNAME = 'valid'
"""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_params_file', type=os.path.abspath,
                        help='Dataset parameters json file path')
    parser.add_argument('interval_params_file', type=os.path.abspath,
                        help='Interval parameters json file path')
    parser.add_argument('model_params_file', type=os.path.abspath,
                        help='Model parameters json file path')
    parser.add_argument('logdir', type=os.path.abspath,
                        help='Log directory, also used as globally unique run identifier')
    parser.add_argument('--visiblegpus', type=str,
                        required=True, help='Visible GPUs string')
    args = parser.parse_args()

    train_tf_dragonn(args.dataset_params_file, args.interval_params_file,
                     args.model_params_file, args.logdir, args.visiblegpus)


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
    for fname in [datasetspec, intervalspec, modelspec]:
        shutil.copyfile(fname, os.path.join(logdir, ntpath.basename(fname)))
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


if __name__ == '__main__':
    main()
