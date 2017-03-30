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

IN_MEMORY = False
BATCH_SIZE = 256
EPOCH_SIZE = 2500000
LEARNING_RATE = 0.0003

# TF Session Settings
DEFER_DELETE_SIZE = int(250 * 1e6)  # 250MB
GPU_MEM_PROP = 0.45  # Allows 2x sessions / gpu


logger = loggers.get_logger('train-wrapper')

backend = K.backend()
if backend != 'tensorflow':
    raise ValueError(
        'Only the keras tensorflow backend is supported, currently using {}'.format(backend))


def parse_args(args):
    parser = argparse.ArgumentParser('TF-DragoNN model runner')

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
    parser.add_argument('--max_examples', type=int, help='max number of examples')

    args = parser.parse_args(args)
    return args


def validate_paths(dataspec, intervalspec, modelspec, logdir):
    for specfile in [dataspec, intervalspec, modelspec]:
        validate_specfile(specfile)
    if os.path.isdir(logdir):  # remove empty directories for debugging
        if len(os.listdir(logdir)) == 0:
            shutil.rmtree(logdir)
    assert(not os.path.exists(logdir))
    assert(logdir.startswith(LOGDIR_PREFIX))


def validate_specfile(specfile):
    specfile = os.path.abspath(specfile)
    assert(os.path.isfile(specfile))
    assert(specfile.startswith(DIR_PREFIX))


def train_tf_dragonn(datasetspec, intervalspec, modelspec, logdir, visiblegpus):
    datasetspec = os.path.abspath(datasetspec)
    intervalspec = os.path.abspath(intervalspec)
    modelspec = os.path.abspath(modelspec)
    logdir = os.path.abspath(logdir)
    os.makedirs(logdir)

    run_id = str(logdir.lstrip(LOGDIR_PREFIX))

    logger.info('dataspec file: {}'.format(datasetspec))
    logger.info('intervalspec file: {}'.format(intervalspec))
    logger.info('logdir path: {}'.format(logdir))
    logger.info('visiblegpus string: {}'.format(visiblegpus))

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

    logger.info("Setting up genomeflow interface")
    data_interface = genomeflow_interface.GenomeFlowInterface(
        datasetspec, intervalspec, modelspec,
        validation_chroms=VALID_CHROMS, holdout_chroms=HOLDOUT_CHROMS)

    logger.info("shuffle: {}".format(data_interface.shuffle))
    logger.info("pos_sampling_rate: {}".format(
        data_interface.pos_sampling_rate))

    logger.info('Setting up genomeflow queues')
    train_queue = data_interface.get_train_queue()
    validation_queue = data_interface.get_validation_queue()
    # normalized_pos_rate = train_queue.normalized_pos_rate ## TODO (johnny): finish this
    # class_weights = {0: 1,
    #                 1: 1 / normalized_pos_rate}

    logger.info('initializing  model and trainer')
    # jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    # with jit_scope():
    model = models.model_from_minimal_config(
        modelspec, train_queue.output_shapes, len(data_interface.task_names))
    loggers.setup_logger('trainer', os.path.join(logdir, "metrics.log"))
    trainer_logger = logging.getLogger('trainer')
    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names,
                                         optimizer='adam',
                                         lr=LEARNING_RATE,
                                         batch_size=BATCH_SIZE,
                                         epoch_size=EPOCH_SIZE,
                                         num_epochs=100,
                                         early_stopping_metric=EARLYSTOPPING_KEY,
                                         early_stopping_patience=EARLYSTOPPING_PATIENCE,
                                         logger=trainer_logger)
    logger.info('training model')
    trainer.train(model, train_queue, validation_queue,
                  save_best_model_to_prefix=os.path.join(logdir, "model"))

    # copy datasetspec, intervalspec, and models params to log dir
    shutil.copyfile(datasetspec, os.path.join(
        logdir, ntpath.basename('datasetspec.json')))
    shutil.copyfile(intervalspec, os.path.join(
        logdir, ntpath.basename('intervalspec.json')))
    shutil.copyfile(modelspec, os.path.join(
        logdir, ntpath.basename('modelspec.json')))


def test_tf_dragonn(logdir, visiblegpus, test_size=None):
    """Currently this tests on *all chroms* (no holdout)"""
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
        datasetspec, intervalspec, modelspec)
    validation_queue = data_interface.get_validation_queue()

    logger.info('loading  model and trainer')
    model = models.model_from_config_and_queue(modelspec, validation_queue)
    model.load_weights(os.path.join(logdir, 'model.weights.h5'))
    trainer = trainers.ClassifierTrainer(task_names=data_interface.task_names)

    logger.info('testing model')
    classification_result = trainer.test(
        model, validation_queue, test_size=test_size)
    logger.info(classification_result)


def predict_tf_dragonn(datasetspec, intervalspec, logdir, visiblegpus, flank_size, prefix):
    datasetspec = os.path.abspath(datasetspec)
    assert(os.path.isfile(datasetspec))
    assert(datasetspec.startswith(DIR_PREFIX))

    intervalspec = os.path.abspath(intervalspec)
    assert(os.path.isfile(intervalspec))
    assert(intervalspec.startswith(DIR_PREFIX))

    logdir = os.path.abspath(logdir)
    assert(os.path.exists(logdir))
    assert(logdir.startswith(LOGDIR_PREFIX))

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
        datasetspec, intervalspec, modelspec, validation_chroms=HOLDOUT_CHROMS, holdout_chroms=[])
    example_queues = {dataset_id: data_interface.get_example_queue(dataset_values, dataset_id,
                                                                   num_epochs=1,
                                                                   input_names=data_interface.input_names,
                                                                   enqueues_per_thread=[128, 1])
                      for dataset_id, dataset_values in data_interface.validation_dataset.items()}

    logger.info('loading  model and trainer')
    model = models.model_from_minimal_config(modelspec,
                                             example_queues.values()[
                                                 0].output_shapes,
                                             len(data_interface.task_names))
    model.load_weights(os.path.join(logdir, 'model.weights.h5'))
    trainer = trainers.ClassifierTrainer()

    def generate_intervals(chroms, starts, ends, preds):
        for chrom, start, end, pred in zip(chroms, starts, ends, preds):
            yield pybedtools.create_interval_from_list([chrom, start, end, str(pred)])

    for dataset_id, example_queue in example_queues.items():
        logger.info('generating predictions for dataset {}'.format(dataset_id))
        intervals, predictions = trainer.predict(model, example_queue)

        # trim flanks
        intervals['start'] += flank_size
        intervals['end'] -= flank_size

        # write each task to bedtool and save
        for task_indx, task_name in enumerate(data_interface.task_names):
            intervals = generate_intervals(intervals['chrom'],
                                           intervals['start'],
                                           intervals['end'],
                                           predictions[:, task_indx])
            bedtool = pybedtools.BedTool(intervals)
            output_fname = "{}.{}.{}.tab.gz".format(
                prefix, task_name, dataset_id)
            bedtool.sort().saveas(output_fname)
            logger.info("\nSaved {} predictions in dataset {} to {}".format(
                task_name, dataset_id, output_fname))
    logger.info('Done!')
