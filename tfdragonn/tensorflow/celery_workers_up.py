#!/usr/bin/env python

import subprocess

import argparse
import socket
import signal
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int)
parser.add_argument('num_replicas', type=int)
args = parser.parse_args()

workers = []
hostname = socket.gethostname()

for gpu_idx in range(args.num_gpus):
    for replica_idx in range(args.num_replicas):
        env = os.environ.copy()
        env['TFDRAGONN_GPU'] = str(gpu_idx)
        name = '{}-gpu-{}-replica-{}'.format(hostname, gpu_idx, replica_idx)
        print('launching {}'.format(name))
        cmd = 'TFDRAGONN_GPU={} celery -A tfdragonn worker -l INFO -c 1 -n {}'.format(
            gpu_idx, name)
        p = subprocess.Popen(cmd, shell=True, env=env)
        workers.append(p)


def signal_handler(signal, frame):
    print('\nstopping celery workers...\n')
    for p in workers:
        p.kill()
    print('\nfinsihed stopping celery workers. Exiting.\n')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
print('Finished workers setup; awaiting sigint')
signal.pause()
