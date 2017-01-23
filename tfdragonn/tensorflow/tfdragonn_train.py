#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from tfdragonn import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_params_file', type=os.path.abspath)
    parser.add_argument('interval_params_file', type=os.path.abspath)
    parser.add_argument('model_params_file', type=os.path.abspath)
    parser.add_argument('logdir', type=str)
    args = parser.parse_args()

    submit_train_task(args.dataset_params_file, args.interval_params_file,
                      args.model_params_file, args.logdir)


def submit_train_task(dataset_params_file, interval_params_file, model_params_file, logdir):
    assert(os.path.isfile(dataset_params_file))
    assert(os.path.isfile(interval_params_file))
    assert(os.path.isfile(model_params_file))
    assert(not os.path.exists(logdir))
    train.delay(dataset_params_file, interval_params_file,
                model_params_file, logdir)

if __name__ == '__main__':
    main()
