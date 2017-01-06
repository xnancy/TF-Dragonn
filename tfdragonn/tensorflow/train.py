#!/usr/bin/env python

import argparse
import csv
import os
import logging
import shutil
import math


import tensorflow as tf

from dataset_interval_reader import get_train_readers_and_tasknames
from dataset_interval_reader import get_valid_readers_and_tasknames
from shared_examples_queue import SharedExamplesQueue
from shared_examples_queue import ValidationSharedExamplesQueue
import models
from trainers import ClassiferTrainer
from early_stopper import train_until_earlystop

HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
VALID_CHROMS = ['chr9']

TRAIN_DIRNAME = 'train'
VALID_DIRNAME = 'valid'

EARLYSTOPPING_KEY = 'metrics/auPRC'
EARLYSTOPPING_PATIENCE = 4
EARLYSTOPPING_TOLERANCE = 1e-4

IN_MEMORY = False
BATCH_SIZE = 128
EPOCH_SIZE = 1000

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('train-wrapper')

parser = argparse.ArgumentParser()
parser.add_argument('--datasetspec', type=str, required=True, help='Dataspec file')
parser.add_argument('--intervalspec', type=str, required=True, help='Intervalspec file')
parser.add_argument('--model-type', type=str, default='SequenceAndDnaseClassifier',
                    help="""Which model to use.
                            Supported options: SequenceAndDnaseClassifier, SequenceDnaseAndDnasePeaksCountsClassifier.
                            Default: SequenceAndDnaseClassifier""")
parser.add_argument('--logdir', type=str, required=True, help='Logging directory')
parser.add_argument('--visiblegpus', type=str, required=True, help='Visible GPUs string')
parser.add_argument('--tasks', type=str, required=False, help='List of tasks as a csv string')
args = parser.parse_args()

assert(os.path.isfile(args.datasetspec))
assert(os.path.isfile(args.intervalspec))

if os.path.isdir(args.logdir):  # remove empty directories for debugging
    if len(os.listdir(args.logdir)) == 0:
        shutil.rmtree(args.logdir)
assert(not os.path.exists(args.logdir))
os.mkdir(args.logdir)
assert(os.path.isdir(args.logdir))
train_log_dir = os.path.join(args.logdir, TRAIN_DIRNAME)
valid_log_dir = os.path.join(args.logdir, VALID_DIRNAME)

tasks = None
if 'tasks' in args:
    if args.tasks:
        tasks = list(csv.reader([args.tasks]))[0]
        logging.info('tasks: {}'.format(tasks))

logging.info('dataspec file: {}'.format(args.datasetspec))
logging.info('intervalspec file: {}'.format(args.intervalspec))
logging.info('logdir path: {}'.format(args.logdir))
logging.info('visiblegpus string: {}'.format(args.visiblegpus))


logging.info('Setting up readers')


def get_model(num_tasks):
    model_class = getattr(models, args.model_type)
    return model_class(num_tasks=num_tasks)


session_config = tf.ConfigProto()
session_config.gpu_options.deferred_deletion_bytes = int(250 * 1e6)  # 250MB
session_config.gpu_options.visible_device_list = args.visiblegpus

trainer = ClassiferTrainer(epoch_size=EPOCH_SIZE)


def train(checkpoint=None, num_epochs=1):
    with tf.Graph().as_default():
        train_readers, task_names = get_train_readers_and_tasknames(
            args.datasetspec, args.intervalspec, validation_chroms=VALID_CHROMS,
            holdout_chroms=HOLDOUT_CHROMS, in_memory=IN_MEMORY, tasks=tasks)
        train_queue = SharedExamplesQueue(train_readers, task_names, batch_size=BATCH_SIZE)
        num_tasks = len(task_names)
        train_model = get_model(num_tasks)

        new_checkpoint = trainer.train(
            train_model, train_queue, train_log_dir, checkpoint, session_config, num_epochs)
        return new_checkpoint


def validate(checkpoint):
    with tf.Graph().as_default():
        valid_readers, task_names, num_valid_exs = get_valid_readers_and_tasknames(
            args.datasetspec, args.intervalspec, validation_chroms=VALID_CHROMS,
            holdout_chroms=HOLDOUT_CHROMS, in_memory=IN_MEMORY, tasks=tasks)
        valid_queue = ValidationSharedExamplesQueue(
            valid_readers, task_names, batch_size=BATCH_SIZE)
        num_batches = int(math.floor(num_valid_exs / BATCH_SIZE) - 1)
        num_tasks = len(task_names)
        valid_model = get_model(num_tasks)

        eval_metrics = trainer.evaluate(
            valid_model, valid_queue, num_batches, valid_log_dir, checkpoint, session_config)
        return eval_metrics


train_until_earlystop(
    train, validate, metric_key=EARLYSTOPPING_KEY, patience=EARLYSTOPPING_PATIENCE,
    tolerance=EARLYSTOPPING_TOLERANCE, max_epochs=100)
