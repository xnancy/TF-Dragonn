from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os
import time
import subprocess

import pandas as ps

MODELS_PER_GPU = 2
NUM_GPUS = 4

WAIT_INTERVAL_SECS = 5

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('train-wrapper')

parser = argparse.ArgumentParser()
parser.add_argument('taskdef_file', help='task definition file', type=str)
parser.add_argument('output_dir', help='output directory', type=str)
args = parser.parse_args()

assert(not os.path.exists(args.output_dir))
os.mkdir(args.output_dir)

tasks = ps.read_csv(args.taskdef_file, sep='\t',
                    names=['name', 'tasks', 'dataset_file', 'intervals_file', 'model_type']).set_index('name')

logger.info('Read {} tasks'.format(tasks.shape[0]))

# Check task names are non-redundent
assert(len(set(tasks.index)) == tasks.shape[0])

# list each GPU multiple times
free_gpus = list(range(NUM_GPUS)) * MODELS_PER_GPU

running_processes = []
commands = []

for (name, tasks, dataset_file, intervals_file, model_type) in tasks.itertuples():
    task_base_dir = os.path.join(args.output_dir, name)
    os.mkdir(task_base_dir)
    task_output_dir = os.path.join(task_base_dir, 'out')
    command = ['python', 'train.py', '--datasetspec', dataset_file, '--intervalspec',
               intervals_file, '--model-type', model_type, '--logdir', task_output_dir,
               '--tasks', tasks]
    commands.append((name, command, task_base_dir))

logger.info('launching trainer processes')

while len(commands):
    while not free_gpus:
        stopped = None
        for i, (name, proc, gpu, out, err) in enumerate(running_processes):
            if proc.poll() is not None:
                stopped = running_processes.pop(i)
                break
        if stopped is None:
            logger.info('...waiting {} seconds for processeses to finish...'.format(
                WAIT_INTERVAL_SECS))
            time.sleep(WAIT_INTERVAL_SECS)
        else:
            (name, proc, gpu, out, err) = stopped
            out.close()
            err.close()
            logger.info('process {} finished with code {}'.format(
                name, proc.returncode))
            free_gpus.append(gpu)

    (name, command, outdir) = commands.pop()
    logger.info('launching process for {}'.format(name))
    gpu = free_gpus.pop()
    command += ['--visiblegpus', str(gpu)]
    out = open(os.path.join(outdir, 'stdout.log.txt'), 'w')
    err = open(os.path.join(outdir, 'stderr.log.txt'), 'w')
    proc = subprocess.Popen(command, stdout=out, stderr=err)
    running_processes.append((name, proc, gpu, out, err))

logger.info('Done launching all proceses; waiting for all to finish')

while len(running_processes):
    stopped = None
    for i, (name, proc, gpu, out, err) in enumerate(running_processes):
        if proc.poll() is not None:
            stopped = running_processes.pop(i)
            break
    if stopped is None:
        logger.info('...waiting {} seconds for {} processeses to finish...'.format(
            WAIT_INTERVAL_SECS, len(running_processes)))
        time.sleep(WAIT_INTERVAL_SECS)
    else:
        (name, proc, gpu, out, err) = stopped
        out.close()
        err.close()
        logger.info('process {} finished with code {}'.format(
            name, proc.returncode))
