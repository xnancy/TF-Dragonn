from __future__ import absolute_import, division, print_function

import argparse
import collections
from functools import partial
import json
import logging
import numpy as np
import os

from genomedatalayer.shm import (
    extract_fasta_to_file, extract_bigwig_to_file,
    extract_bed_counts_to_file, extract_dist_to_intervals_to_file
)

from tfdragonn.datasets import parse_raw_intervals_config_file
from tfdragonn.intervals import get_tf_predictive_setup
from tfdragonn.tensorflow import AMBIG_LABEL
from tfdragonn.tensorflow.datasets import (
    parse_raw_inputs, raw_input2processed_input,
    parse_processed_intervals
)
from tfdragonn.tensorflow.extractors import extract_expression_to_file

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

input2extractor_func = {'dnase_bigwig': extract_bigwig_to_file,
                        'genome_fasta': extract_fasta_to_file,
                        'dnase_peaks_bed': extract_bed_counts_to_file,
                        'gencode_tss': extract_bed_counts_to_file,
                        'gencode_annotation': extract_dist_to_intervals_to_file,
                        'gencode_polyA': extract_dist_to_intervals_to_file,
                        'gencode_lncRNA': extract_dist_to_intervals_to_file,
                        'gencode_tRNA': extract_dist_to_intervals_to_file,
                        'expression_tsv': extract_expression_to_file}

def parse_args():
    parser = argparse.ArgumentParser(
        description='TensorFlow DragoNN (TF-DragoNN) sub-commands include:')
    # define parsers with common arguments
    raw_intervals_config_parser = argparse.ArgumentParser(add_help=False)
    raw_intervals_config_parser.add_argument('--raw-intervals-config-file', type=str, required=True,
                                             help='includes task names and map from dataset ids to raw interval files')
    processed_intervals_config_parser = argparse.ArgumentParser(add_help=False)
    processed_intervals_config_parser.add_argument('--processed-intervals-config-file', type=str, required=True,
                                                   help='includes task names and map from dataset ids to processed interval files')
    multiprocessing_parser = argparse.ArgumentParser(add_help=False)
    multiprocessing_parser.add_argument('--n-jobs', type=int, default=1,
                                        help='num of processes')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    # define commands
    subparsers = parser.add_subparsers(help='tf-dragonn command help', dest='command')
    extract_parser = subparsers.add_parser('extract',
                                          help="""This command extracts encoded data from raw data files in
                                          the data config file for use with streaming models,
                                          and writes a new data config with memmaped inputs.""")
    extract_parser.add_argument('--extract-dir', type=str, required=True,
                               help='directories with extracted data are created in this directory.')
    extract_parser.add_argument('--raw-inputs-config-file', type=str, required=True, help="Raw input configuration file.")
    extract_parser.add_argument('--processed-inputs-config-file', type=str, required=True, help="Processed input configuration file to write.")
    label_regions_parser = subparsers.add_parser('label_regions',
                                                 parents=[raw_intervals_config_parser,
                                                          multiprocessing_parser,
                                                          prefix_parser],
                                         help='Generates fixed length regions and their labels for each dataset.'
                                              'Writes a new data config file with regions and labels files.')
    label_regions_parser.add_argument('--bin-size', type=int, default=200,
                                       help='size of bins for labeling')
    label_regions_parser.add_argument('--flank-size', type=int, default=400,
                                       help='size of flanks around labeled bins')
    label_regions_parser.add_argument('--stride', type=int, default=50,
                                       help='spacing between consecutive bins')
    subset_parser = subparsers.add_parser('subset',
                                          parents=[processed_intervals_config_parser,
                                                   prefix_parser],
                                           help="""This command subsets processed intervals and labels into individual tasks.""")
    # return command and command arguments
    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def main_extract(raw_inputs_config_file=None,
                 extract_dir=None,
                 processed_inputs_config_file=None):
    """
    Extracts data for every raw input in the data config file.
    Returns new data config file with extracted inputs.
    """
    import ntpath
    dataset_id2raw_inputs = parse_raw_inputs(raw_inputs_config_file)
    dataset_id2processed_inputs = collections.OrderedDict()
    logger.info("Processing input data in {}...".format(raw_inputs_config_file))
    for dataset_id, raw_inputs in dataset_id2raw_inputs.items():
        processed_inputs = {}
        for input_name, input_vals in raw_inputs.items():
            if input_vals is None:
                processed_inputs[raw_input2processed_input[input_name]] = None
                continue
            extractor_func = input2extractor_func[input_name]
            if isinstance(input_vals, (str, unicode)): # only datafile is specified
                datafile = input_vals
                extract_subdir = datafile
            else: # datafile and extraction args are specified
                datafile = input_vals['datafile']
                del input_vals['datafile']
                if 'extract_subdir' in input_vals:
                    extract_subdir = input_vals['extract_subdir']
                    del input_vals['extract_subdir']
                else:
                    extract_subdir = datafile
                extractor_func = partial(extractor_func, **input_vals)
            input_extract_dir = os.path.join(extract_dir, ntpath.basename(extract_subdir))
            logger.info("Extracting data from {} into {}...".format(datafile, input_extract_dir))    
            try:
                extractor_func(datafile, input_extract_dir)
            except OSError:
                logger.info("{} already exists.".format(input_extract_dir))
                pass
            processed_inputs[raw_input2processed_input[input_name]] = input_extract_dir
        dataset_id2processed_inputs[dataset_id] = processed_inputs
    # write json with extracted data
    json.dump(dataset_id2processed_inputs, open(processed_inputs_config_file, "w"), indent=4)
    logger.info("Wrote processed inputs config file to {}.".format(processed_inputs_config_file))
    logger.info("Done!")


def main_label_regions(raw_intervals_config_file=None,
                       bin_size=None,
                       flank_size=None,
                       stride=None,
                       n_jobs=None,
                       prefix=None):
    """
    Generates regions and labels files for each dataset.
    Writes new data config file with the generated files.
    """
    raw_intervals_config = parse_raw_intervals_config_file(raw_intervals_config_file)
    processed_intervals_dict = collections.OrderedDict([("task_names", raw_intervals_config.task_names)])
    logger.info("Generating regions and labels for datasets in {}...".format(raw_intervals_config_file))
    for dataset_id, raw_intervals in raw_intervals_config:
        logger.info("Generating regions and labels for dataset {}...".format(dataset_id))
        dataset_prefix = "{}.{}".format(prefix, dataset_id)
        path_to_dataset_regions = os.path.abspath("{}.intervals.bed".format(dataset_prefix))
        path_to_dataset_labels = os.path.abspath("{}.labels.npy".format(dataset_prefix))
        if os.path.isfile("{}.intervals.bed".format(dataset_prefix)) and os.path.isfile("{}.labels.npy".format(dataset_prefix)):
            logger.info("Regions file {} and labels file {} already exist. skipping dataset {}!".format(path_to_dataset_regions, path_to_dataset_labels, dataset_id))
        else:
            regions, labels = get_tf_predictive_setup(raw_intervals.feature_beds, region_bedtool=raw_intervals.region_bed,
                                                      ambiguous_feature_bedtools=raw_intervals.ambiguous_feature_beds,
                                                      bin_size=bin_size, flank_size=flank_size, stride=stride,
                                                      filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                      save_to_prefix=dataset_prefix)
            logger.info("Saved regions to {} and labels to {}".format(path_to_dataset_regions, path_to_dataset_labels))
        processed_intervals_dict[dataset_id] = {"regions": path_to_dataset_regions, "labels": path_to_dataset_labels}
    # write processed intervals config file
    processed_intervals_config_file = os.path.abspath("{}.json".format(prefix))
    json.dump(processed_intervals_dict, open(processed_intervals_config_file, "w"), indent=4)
    logger.info("Wrote new data config file to {}.".format(processed_intervals_config_file))
    logger.info("Done!")


def main_subset(processed_intervals_config_file=None,
                prefix=None):
    """
    Subsets a processed intervals config file by tasks.
    """
    logger.info("Splitting data in {} into individual tasks...".format(processed_intervals_config_file))
    processed_intervals_dict = parse_processed_intervals(processed_intervals_config_file)
    task_names = processed_intervals_dict['task_names']
    task2task_config = collections.OrderedDict()
    for task_name in task_names: # initialize task configs with task_names
        task2task_config[task_name] = collections.OrderedDict([("task_names", [task_name])])
    for dataset_id, dataset_dict in processed_intervals_dict.items():
        if dataset_id == "task_names":
            continue
        logger.info("Splitting dataset {}...".format(dataset_id))
        dataset_prefix = "{}.{}".format(prefix, dataset_id)
        for task_indx, task_name in enumerate(task_names):
            path_to_labels = os.path.abspath("{}.{}.labels.npy".format(dataset_prefix, task_name))
            if os.path.isfile(path_to_labels):
                task2task_config[task_name][dataset_id] = {"regions": dataset_dict['regions'], "labels": path_to_labels}
                logger.info("Labels file {} already exist for task {} in dataset {}!".format(
                    path_to_labels, task_name, dataset_id))
                continue
            task_labels = dataset_dict['labels'][:, task_indx][:, None]
            if np.all(task_labels == AMBIG_LABEL): # no data for this task
                logger.info("No data for task {} in dataset {}, proceeding to next task!".format(task_name, dataset_id))
                continue
            np.save(path_to_labels, task_labels)
            task2task_config[task_name][dataset_id] = {"regions": dataset_dict['regions'], "labels": path_to_labels}
            logger.info("Saved labels for {} to {}".format(dataset_id, path_to_labels))
    # write task config files
    for task_name, task_config in task2task_config.items():
        config_fname = os.path.abspath("{}.{}.json".format(prefix, task_name))
        json.dump(task_config, open(config_fname, "w"), indent=4)
        logger.info("Wrote {} config file to {}.".format(task_name, config_fname))
    logger.info("Done!")


def main():
    # parse args
    command_functions = {'extract': main_extract,
                         'label_regions': main_label_regions,
                         'subset': main_subset}
    command, args = parse_args()
    # run command
    command_functions[command](**args)
