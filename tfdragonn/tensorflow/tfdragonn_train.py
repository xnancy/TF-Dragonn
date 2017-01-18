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

    assert(os.path.isfile(args.dataset_params_file))
    assert(os.path.isfile(args.interval_params_file))
    assert(os.path.isfile(args.model_params_file))
    assert(not os.path.exists(args.logdir))

    train.delay(args.dataset_params_file, args.interval_params_file,
          args.model_params_file, args.logdir)


if __name__ == '__main__':
    main()
