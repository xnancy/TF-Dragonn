from __future__ import absolute_import

import argparse
import json
import logging
from memory_profiler import profile
import numpy as np
import os
from pybedtools import BedTool

from genomedatalayer.extractors import (
    FastaExtractor, MemmappedBigwigExtractor, MemmappedFastaExtractor
)

from tfdragonn.datasets import Dataset, parse_data_config_file
from tfdragonn.intervals import get_tf_predictive_setup, train_test_chr_split 

# setup logging
log_formatter = \
    logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
logger = logging.getLogger('tf-dragonn')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

RAW_INPUT_KEYS = ['dnase_bigwig', 'genome_fasta']
input2memmap_extractor = {'dnase_bigwig': MemmappedBigwigExtractor,
                          'genome_fasta': MemmappedFastaExtractor}
input2memmap_input = {'dnase_bigwig': 'dnase_data_dir',
                      'genome_fasta': 'genome_data_dir'}

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for DragoNN modeling of TF Binding.')
    # define parsers with common arguments
    data_config_parser = argparse.ArgumentParser(add_help=False)
    data_config_parser.add_argument('--data-config-file', type=str, required=True,
                                   help='file with dataset names, task names, and data files')
    genome_fasta_parser = argparse.ArgumentParser(add_help=False)
    genome_fasta_parser.add_argument('--genome-fasta', type=str, required=True,
                                   help='Genome fasta file')
    multiprocessing_parser = argparse.ArgumentParser(add_help=False)
    multiprocessing_parser.add_argument('--n-jobs', type=int, default=1,
                                        help='num of processes used for preprocessing and postprocessing')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    num_intervals_parser = argparse.ArgumentParser(add_help=False)
    num_intervals_parser.add_argument('--num-intervals', type=int,
                                      help='caps number of intervals used')
    memmap_dir_parser = argparse.ArgumentParser(add_help=False)
    memmap_dir_parser.add_argument('--memmap-dir', type=str, required=True,
                               help='directories with memmaped data are created in this directory.')
    
    # define commands 
    subparsers = parser.add_subparsers(help='tf-dragonn command help', dest='command')
    sequences_parser = subparsers.add_parser('sequences',
                                             parents=[data_config_parser,
                                                      multiprocessing_parser,
                                                      num_intervals_parser,
                                                      prefix_parser],
                                             help='testing memory usage of sequences batch generation')
    sequence_and_label_streaming_parser = subparsers.add_parser('sequence_and_label_streaming',
                                                                parents=[data_config_parser,
                                                                         genome_fasta_parser,
                                                                         multiprocessing_parser,
                                                                         num_intervals_parser,
                                                                         prefix_parser],
                                                                help='testing correctness of sequence and label streaming')
    # return command and command arguments
    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def get_regions_and_labels(data_config_file=None,
                           prefix=None,
                           n_jobs=None,
                           **kwargs):
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    datasets = parse_data_config_file(data_config_file)
    # get regions and labels, split into train/test
    if len(datasets) > 1:
        raise RuntimeError("Data configurations with more than one datasets are not supported yet!")
    else:
        data = datasets[datasets.keys()[0]]
        if data.genome_data_dir is None:
            raise ValueError("genome_data_dir is required for this test!")
    if os.path.isfile("{}.intervals.bed".format(prefix)) and os.path.isfile("{}.labels.npy".format(prefix)):
        logger.info("loading intervals from {}.intervals.bed".format(prefix))
        regions = BedTool("{}.intervals.bed".format(prefix))
        logger.info("loading labels from {}.labels.npy".format(prefix))
        labels = np.load("{}.labels.npy".format(prefix))
    elif data['regions'] is not None and data['labels'] is not None:
        if os.path.isfile(data['regions']) and os.path.isfile(data['labels']):
            logger.info("loading intervals from {}".format(data['regions']))
            regions = BedTool(data['regions'])
            logger.info("loading labels from {}".format(data['labels']))
            labels = np.load(data['labels'])
    else:
        logger.info("getting regions and labels")
        regions, labels = get_tf_predictive_setup(data.feature_beds, region_bedtool=data.region_bed,
                                                  bin_size=200, flank_size=400, stride=200,
                                                  filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                  save_to_prefix=prefix)
    return data, regions, labels


def main_sequence_and_label_streaming(data_config_file=None,
                                      genome_fasta=None,
                                      prefix=None,
                                      n_jobs=None,
                                      data=None,
                                      regions=None,
                                      labels=None,
                                      num_intervals=None,
                                      **kwargs):
    from keras.utils.generic_utils import Progbar
    from tfdragonn.io_utils import generate_from_intervals_and_labels

    assert genome_fasta is not None
    assert data.genome_data_dir is not None
    fasta_extractor = FastaExtractor(genome_fasta)
    mm_fasta_extractor = MemmappedFastaExtractor(data.genome_data_dir)
    if num_intervals is not None:
        logger.info("Using total of {} intervals to test sequence and label streaming".format(num_intervals))
        intervals = regions.at(range(num_intervals))
    else:
        logger.info("Using total of {} intervals to test sequence and label streaming".format(len(regions)))
        intervals= regions
    logger.info("loading sequence into memory...")
    X_in_memory = fasta_extractor(intervals)
    logger.info("Streaming sequences and labels and comparing to data in memory...")
    samples_per_epoch = len(intervals)
    batch_size=128
    batches_per_epoch = samples_per_epoch / batch_size + 1
    batch_array = np.zeros((batch_size, 1, 4, intervals[0].length), dtype=np.float32)
    batch_generator = generate_from_intervals_and_labels(intervals, labels, mm_fasta_extractor,
                                                         batch_size=batch_size, indefinitely=False, batch_array=batch_array)
    progbar = Progbar(target=samples_per_epoch)
    for batch_indx in xrange(1, batches_per_epoch + 1):
        X_batch, labels_batch  = next(batch_generator)
        start = (batch_indx-1)*batch_size
        stop = batch_indx*batch_size
        if stop > samples_per_epoch:
            stop = samples_per_epoch
        # assert streamed sequences and labels match data in memory 
        assert (X_in_memory[start:stop] - X_batch).sum() == 0
        assert (labels[start:stop] - labels_batch).sum() == 0
        progbar.update(stop)


def main_sequences(data_config_file=None,
                   prefix=None,
                   n_jobs=None,
                   data=None,
                   regions=None,
                   labels=None,
                   num_intervals=None,
                   **kwargs):
    from tfdragonn.io_utils import infinite_batch_iter
    from keras.utils.generic_utils import Progbar
    import psutil
    if num_intervals is not None:
        logger.info("Using total of {} intervals to test sequence batch streaming".format(num_intervals))
        intervals = regions.at(range(num_intervals))
    else:
        logger.info("Using total of {} intervals to test sequence batch streaming".format(len(regions)))
        intervals= regions
    interval_length = intervals[0].length
    # set up extractor and interval generation
    fasta_extractor = MemmappedFastaExtractor(data.genome_data_dir)
    print("starting batch generation...")
    process = psutil.Process(os.getpid())
    samples_per_epoch = 2000000
    batch_size = 128
    batches_per_epoch = samples_per_epoch / batch_size
    out = np.zeros((batch_size, 1, 4, interval_length), dtype=np.float32)
    interval_batch_iterator = infinite_batch_iter(intervals, batch_size)
    progbar = Progbar(target=samples_per_epoch)
    for batch_indxs in xrange(1, batches_per_epoch + 1):
        out = fasta_extractor(next(interval_batch_iterator), out=out)
        progbar.update(batch_indxs*batch_size, values=[("Non-shared RSS (Mb)", (process.memory_info().rss -  process.memory_info().shared)  / 10**6)])
    logger.info("Done!")


def main():
    # parse args
    command_functions = {'sequences': main_sequences,
                         'sequence_and_label_streaming': main_sequence_and_label_streaming}
    command, args = parse_args()
    # get regions and labels
    data, regions, labels = get_regions_and_labels(**args)
    # run command
    command_functions[command](data=data, regions=regions, labels=labels, **args)


if __name__ == "__main__":
    main()
