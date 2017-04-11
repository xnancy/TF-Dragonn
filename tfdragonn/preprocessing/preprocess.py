#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import numpy as np

from tfdragonn import loggers
from tfdragonn.preprocessing.raw_datasets import parse_raw_intervals_config_file
from tfdragonn.preprocssing.intervals import get_tf_predictive_setup


LOGGER_NAME = 'tfdragonn-preprocess'
_logger = loggers.get_logger(LOGGER_NAME)


def parse_args(args):
    parser = argparse.ArgumentParser('Generate fixed length regions and their'
                                     ' labels for each dataset.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('raw_intervals_config_file', type=str,
                        help='Includes task names and a map from dataset id -> raw interval file')
    parser.add_argument('prefix', type=str, help='prefix of output files')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='num of processes.\nDefault: 1.')
    parser.add_argument('--bin-size', type=int, default=200,
                        help='size of bins for labeling.\nDefault: 200.')
    parser.add_argument('--flank-size', type=int, default=400,
                        help='size of flanks around labeled bins.\nDefault: 400.')
    parser.add_argument('--stride', type=int, default=50,
                        help='spacing between consecutive bins.\nDefault: 50.')
    parser.add_argument('--genome', type=str, default='hg19',
                        help='Genome name.\nDefault: hg19.'
                        '\nOptions: hg18, hg38, mm9, mm10, dm3, dm6.')
    parser.add_argument('--logdir', type=os.path.abspath,
                        help='Logging directory', default=None)
    args = parser.parse_args()
    return args


def run_label_regions_from_args(command, args):
    args = parse_args(args)
    label_regions(raw_intervals_config_file=args.raw_intervals_config_file,
                  prefix=args.prefix, n_jobs=args.n_jobs, bin_size=args.bin_size,
                  flank_size=args.flank_size, genome=args.genome, logdir=args.logdir)


def label_regions(raw_intervals_config_file, prefix,
                  n_jobs=1, bin_size=200, flank_size=400, stride=50, genome='hg19', logdir=None):
    """Generate regions and labels files for each dataset.

    Writes new data config file with the generated files.
    """
    if logdir is not None:
        loggers.add_logdir(LOGGER_NAME, logdir)
    raw_intervals_config = parse_raw_intervals_config_file(
        raw_intervals_config_file)
    processed_intervals_dict = collections.OrderedDict(
        [("task_names", raw_intervals_config.task_names)])
    _logger.info("Generating regions and labels for datasets in {}...".format(
        raw_intervals_config_file))
    for dataset_id, raw_intervals in raw_intervals_config:
        _logger.info(
            "Generating regions and labels for dataset {}...".format(dataset_id))
        path_to_dataset_intervals_file = os.path.abspath(
            "{}.{}.intervals_file.tsv.gz".format(prefix, dataset_id))
        if os.path.isfile(path_to_dataset_intervals_file):
            _logger.info("intervals_file file {} already exists. skipping dataset {}!".format(
                path_to_dataset_intervals_file, dataset_id))
        else:
            intervals, labels = get_tf_predictive_setup(raw_intervals.feature_beds, region_bedtool=raw_intervals.region_bed,
                                                        ambiguous_feature_bedtools=raw_intervals.ambiguous_feature_beds,
                                                        bin_size=bin_size, flank_size=flank_size, stride=stride,
                                                        filter_flank_overlaps=False, genome=genome, n_jobs=n_jobs)
            intervals_file_array = np.empty(
                (labels.shape[0], 3 + labels.shape[1]), np.dtype((str, 10)))
            intervals_file_array[:, :3] = intervals.to_dataframe().as_matrix()[
                :, :3]
            intervals_file_array[:, 3:] = labels
            # np.save(path_to_dataset_intervals_file, intervals_file_array)
            np.savetxt(path_to_dataset_intervals_file,
                       intervals_file_array, delimiter='\t', fmt='%s')
            _logger.info("Saved intervals_file file to {}".format(
                path_to_dataset_intervals_file))
        processed_intervals_dict[dataset_id] = {
            "intervals_file": path_to_dataset_intervals_file}
    # write processed intervals config file
    processed_intervals_config_file = os.path.abspath("{}.json".format(prefix))
    json.dump(processed_intervals_dict, open(
        processed_intervals_config_file, "w"), indent=4)
    _logger.info("Wrote new data config file to {}.".format(
        processed_intervals_config_file))
    _logger.info("Done!")
