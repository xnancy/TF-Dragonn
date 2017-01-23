#!/usr/bin/env python

import subprocess

import argparse
import socket
import signal
import sys

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int)
parser.add_argument('num_replicas', type=int)
args = parser.parse_args()

workers = []
hostname = socket.gethostname()

for gpu_idx in range(args.num_gpus):
    for replica_idx in range(args.num_replicas):
        name = '{}-gpu-{}-replica-{}'.format(hostname, gpu_idx, replica_idx)
        print('launching {}'.format(name))
        p = subprocess.Popen(['TFDRAGONN_GPU={}'.format(
            gpu_idx), 'celery', '-A', 'tfdragonn', 'worker', '-l', 'INFO', '-c', '1', '-n', name])
        workers.append(p)


def signal_handler(signal, frame):
    print('stopping celery_workers_up...')
    for p in workers:
        p.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
print('Finished workers setup; awaiting sigint')
signal.pause()
