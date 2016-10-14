from __future__ import absolute_import

import argparse
import json
import logging
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
    multiprocessing_parser = argparse.ArgumentParser(add_help=False)
    multiprocessing_parser.add_argument('--n-jobs', type=int, default=1,
                                        help='num of processes used for preprocessing and postprocessing')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    num_intervals_parser = argparse.ArgumentParser(add_help=False)
    num_intervals_parser.add_argument('--num-intervals', type=str,
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
                                             help='testing of sequences batch generation')
    # return command and command arguments
    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def main_sequences(data_config_file=None,
                   prefix=None,
                   n_jobs=None,
                   num_intervals=None):
    from tfdragonn.io_utils import infinite_batch_iter
    from keras.utils.generic_utils import Progbar
    import psutil
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
    samples_per_epoch = 5000000
    batch_size = 128
    batches_per_epoch = samples_per_epoch / batch_size
    out = np.zeros((batch_size, 1, 4, intervals[0].length), dtype=np.float32)
    batch_array = np.zeros((batch_size, 1, 4, intervals[0].length), dtype=np.float32)
    interval_batch_iterator = infinite_batch_iter(intervals, batch_size)
    progbar = Progbar(target=samples_per_epoch)
    for batch_indxs in range(1, batches_per_epoch + 1):
        batch_array = fasta_extractor(next(interval_batch_iterator), out=out)
        progbar.update(batch_indxs*batch_size, values=[("RSS (Gb)", process.memory_info().rss / 10**9)])
    logger.info("Done!")


def main():
    # parse args
    command_functions = {'sequences': main_sequences}
    command, args = parse_args()
    # run command
    command_functions[command](**args)


if __name__ == "__main__":
    main()
