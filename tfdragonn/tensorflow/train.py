#!/usr/bin/env python

import argparse
import os
import logging
import shutil


import tensorflow as tf

from dataset_interval_reader import get_train_readers_and_tasknames
from dataset_interval_reader import get_valid_readers_and_tasknames
from shared_examples_queue import SharedExamplesQueue
from shared_examples_queue import ValidationSharedExamplesQueue
from models import SequenceAndDnaseClassifier
from trainers import ClassiferTrainer

HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
VALID_CHROMS = ['chr9']

TRAIN_DIRNAME = 'train'
VALID_DIRNAME = 'valid'

IN_MEMORY = False

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('train-wrapper')

parser = argparse.ArgumentParser()
parser.add_argument('--datasetspec', type=str, required=True, help='Dataspec file')
parser.add_argument('--intervalspec', type=str, required=True, help='Intervalspec file')
parser.add_argument('--logdir', type=str, required=True, help='Logging directory')
parser.add_argument('--visiblegpus', type=str, required=True, help='Visible GPUs string')
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

logging.info('dataspec file: {}'.format(args.datasetspec))
logging.info('intervalspec file: {}'.format(args.intervalspec))
logging.info('logdir path: {}'.format(args.logdir))
logging.info('visiblegpus string: {}'.format(args.visiblegpus))

train_graph = tf.Graph()
valid_graph = tf.Graph()

logging.info('Setting up readers')

with train_graph.as_default():
    train_readers, task_names = get_train_readers_and_tasknames(
        args.datasetspec, args.intervalspec, validation_chroms=VALID_CHROMS,
        holdout_chroms=HOLDOUT_CHROMS, in_memory=IN_MEMORY)
    train_shared_queue = SharedExamplesQueue(train_readers, task_names, batch_size=128)

with valid_graph.as_default():
    valid_readers, task_names = get_valid_readers_and_tasknames(
        args.datasetspec, args.intervalspec, validation_chroms=VALID_CHROMS,
        holdout_chroms=HOLDOUT_CHROMS, in_memory=IN_MEMORY)
    valid_shared_queue = ValidationSharedExamplesQueue(valid_readers, task_names, batch_size=128)

logging.info('Setting up model')


def get_model():
    return SequenceAndDnaseClassifier(num_tasks=num_tasks, fc_layer_widths=(800, 300, 50))

num_tasks = len(task_names)
with train_graph.as_default():
    train_model = get_model()
    trainer = ClassiferTrainer(train_model, num_epochs=1, epoch_size=10000)
with valid_graph.as_default():
    valid_model = get_model()
    evaluator = ClassiferTrainer(valid_model)

logging.info('Starting training')

session_config = tf.ConfigProto()
session_config.gpu_options.deferred_deletion_bytes = int(250 * 1e6)  # 250MB
session_config.gpu_options.visible_device_list = args.visiblegpus

with train_graph.as_default():
    checkpoint = trainer.train(train_shared_queue, train_log_dir, session_config=session_config)

logging.info('Starting eval')
with valid_graph.as_default():
    trainer.evaluate(valid_shared_queue, valid_log_dir, checkpoint, session_config)
