#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
import shutil
import math

import tensorflow as tf
import genomeflow as gf

import database
import models
import datasets

from trainers import ClassiferTrainer
from early_stopper import train_until_earlystop

DIR_PREFIX = '/srv/scratch/tfbinding/'
LOGDIR_PREFIX = '/srv/scratch/tfbinding/tf_logs/'

HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
VALID_CHROMS = ['chr9']

TRAIN_DIRNAME = 'train'
VALID_DIRNAME = 'valid'

EARLYSTOPPING_KEY = 'metrics/auPRC'
EARLYSTOPPING_PATIENCE = 4
EARLYSTOPPING_TOLERANCE = 1e-4

IN_MEMORY = False
BATCH_SIZE = 128
EPOCH_SIZE = 250000

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

    training_dataset, validation_dataset = datasets.parse_inputs_and_intervals(
        datasetspec, intervalspec)

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

    train_log_dir = os.path.join(logdir, TRAIN_DIRNAME)
    valid_log_dir = os.path.join(logdir, VALID_DIRNAME)

    logging.info('dataspec file: {}'.format(datasetspec))
    logging.info('intervalspec file: {}'.format(intervalspec))
    logging.info('logdir path: {}'.format(logdir))
    logging.info('visiblegpus string: {}'.format(visiblegpus))

    logging.info('registering with tfdragonn database')
    metadata = {}  # TODO(cprobert): save metadata here
    database.add_run(run_id, datasetspec, intervalspec,
                     modelspec, logdir, metadata)

    logging.info('Setting up readers')

    model = models.model_from_config(modelspec)

    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = int(
        250 * 1e6)  # 250MB
    session_config.gpu_options.visible_device_list = str(visiblegpus)
    # allows 2 sessions/GPU
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45

    trainer = ClassiferTrainer(epoch_size=EPOCH_SIZE)

    with tf.Graph().as_default():
        with tf.variable_scope('GenomeDataIO'):
            examples_queues = {}
            for dataset_id, dataset_fields in training_dataset.items():

                intervals = dataset_fields['intervals']
                inputs = dataset_fields['inputs']
                labels = dataset_fields['labels']
                task_names = dataset_fields['task_names']

                interval_queue = gf.io.IntervalQueue(
                    intervals, labels, name='{}-interval-queue'.format(dataset_id),
                    capacity=10000, shuffle=False, summary=True)

                data_sources = {}
                for data_type, data_path in inputs.items():
                    if data_type in {'genome_data_dir', 'dnase_data_dir'}:
                        data_sources[data_type] = gf.io.DataSource(data_path, 'bcolz')
                    else:
                        data_sources[data_type] = gf.io.DataSource(
                            data_path, 'bed',
                            {'op': 'max', 'window_half_widths': [1000, 10000]})

                examples_queues[dataset_id] = gf.io.ExampleQueue(
                    interval_queue, data_sources, num_threads=1,
                    enqueue_batch_size=128, capacity=2048,
                    name='{}-example-queue'.format(dataset_id))

            shared_examples_queue = gf.io.MultiDatasetExampleQueue(
                examples_queues, num_threads=1, enqueue_batch_size=128,
                capacity=2048, name='multi-dataset-example-queue')

            examples = shared_examples_queue.dequeue_many(BATCH_SIZE)



    def train(checkpoint=None, num_epochs=1):
        new_checkpoint = trainer.train(
            model, train_queue, train_log_dir, checkpoint, session_config, num_epochs)
        return new_checkpoint

    def validate(checkpoint):

            eval_metrics = trainer.evaluate(
                model, valid_queue, num_batches, valid_log_dir, checkpoint, session_config)
            return eval_metrics

    train_until_earlystop(
        train, validate, metric_key=EARLYSTOPPING_KEY, patience=EARLYSTOPPING_PATIENCE,
        tolerance=EARLYSTOPPING_TOLERANCE, max_epochs=100)


if __name__ == '__main__':
    main()
