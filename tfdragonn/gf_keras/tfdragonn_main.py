#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import model_runner
import preprocessing


command_functions = {
    'train': model_runner.run_from_args,
    'test': model_runner.run_from_args,
    'predict': model_runner.run_from_args,  # TODO: make a predict function
    'label_regions': preprocessing.label_regions,
}
commands_str = ', '.join(command_functions.keys())


def parse_args():
    parser = argparse.ArgumentParser('main TF-DragoNN script')
    parser.add_argument('command', type=str,
                        help='TF-DragoNN command. Must be one of: {}'.format(commands_str))
    args, unknown = parser.parse_known_args()
    return args.command, unknown


def main():
    command, command_args = parse_args()
    if command not in command_functions:
        raise NameError('Command `{}` was not found. Possible commands: {}'.format(
            command, commands_str))
    command_fn = command_functions[command]
    command_fn(command, command_args)


if __name__ == '__main__':
    main()
