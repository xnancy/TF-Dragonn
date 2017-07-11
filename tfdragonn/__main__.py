#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tfdragonn.model_runner
import tfdragonn.preprocessing.preprocess


command_functions = {
    'train': tfdragonn.model_runner.TrainRunner().run_from_args,
    'test': tfdragonn.model_runner.TestRunner().run_from_args,
    'predict': tfdragonn.model_runner.PredictRunner().run_from_args,  # TODO: make a predict module
    'labelregions': tfdragonn.preprocessing.preprocess.run_label_regions_from_args,
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='TF-DragoNN command line tools',
    usage='''tfdragonn <command> <args>

    The tfdragonn commands are:
    train           Train a model
    test            Test a model
    predict         Run prediction on a list of regions
    labelregions    Label a list of regions for training
    ''')
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))


def main():
    args = parser.parse_args(sys.argv[1:2])
    if args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    command_fn = command_functions[args.command]
    command_fn(args.command, sys.argv[2:])


if __name__ == '__main__':
    main()
