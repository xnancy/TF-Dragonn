#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import yaml

from celery import Celery


# from train import train_tf_dragonn

"""
A Celery worker app for training a tf-dragonn model.
"""

CELERY_CONFIG_FILE = 'tfdragonn_celery_queue_config.yaml'


def get_gpu():
    GPU_TO_USE = os.environ.get('TFDRAGONN_GPU')
    if GPU_TO_USE is None:
        raise ValueError('GPU to use is not valid: {}'.format(GPU_TO_USE))
    else:
        GPU_TO_USE = int(GPU_TO_USE)
    return GPU_TO_USE

if not os.path.isfile(CELERY_CONFIG_FILE):
    raise FileNotFoundError(
        'Celery config file does not exist: {}'.format(CELERY_CONFIG_FILE))
with open(CELERY_CONFIG_FILE, 'r') as fp:
    broker = yaml.load(fp)

app = Celery('tfdragonn', broker=broker)


@app.task
def train(dataset_params_file, interval_params_file, model_params_file, logdir):
    print("running task on gpu: {}".format(get_gpu()))
    time.sleep(3)
    print((dataset_params_file, interval_params_file, model_params_file, logdir))
    # train_tf_dragonn(dataset_params_file, interval_params_file,
    #                  model_params_file, logdir, args.visiblegpus, args.tasks)
