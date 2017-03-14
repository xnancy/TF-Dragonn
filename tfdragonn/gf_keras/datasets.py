from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json

import numpy as np
from pybedtools import BedTool

RAW_INPUT_NAMES = set(['genome_fasta', 'dnase_bigwig', 'dnase_peaks_bed',
                       'gencode_tss', 'gencode_annotation', 'gencode_polyA',
                       'gencode_lncRNA', 'gencode_tRNA',
                       'expression_tsv'])

raw_input2processed_input = {'genome_fasta': 'genome_data_dir',
                             'dnase_bigwig': 'dnase_data_dir',
                             'dnase_peaks_bed': 'dnase_peaks_counts_data_dir',
                             'gencode_tss': 'gencode_tss_peaks_counts_data_dir',
                             'gencode_annotation': 'gencode_annotation_distances_data_dir',
                             'gencode_polyA': 'gencode_polyA_distances_data_dir',
                             'gencode_lncRNA': 'gencode_lncRNA_distances_data_dir',
                             'gencode_tRNA': 'gencode_tRNA_distances_data_dir',
                             'expression_tsv': 'expression_data_dir'}

PROCESSED_INPUT_NAMES = set(['genome_data_dir', 'dnase_data_dir',
                             'tss_counts', 'dhs_counts',
                             'tss_mean_tpm', 'tss_max_tpm',
                             'HelT_data_dir', 'MGW_data_dir', 'OC2_data_dir',
                             'ProT_data_dir', 'Roll_data_dir'])


def parse_inputs_and_intervals(processed_inputs_file, processed_intervals_file):
    """Parse the processed inputs and intervals files, return a dataset dictionary."""

    all_datasets = {}
    with open(processed_inputs_file, 'r') as fp:
        data = json.load(fp)
    for cur_dataset_id, cur_dataset_dict in data.items():
        all_datasets[cur_dataset_id] = {}
        for input_id in PROCESSED_INPUT_NAMES:
            if input_id in cur_dataset_dict:
                all_datasets[cur_dataset_id][
                    input_id] = cur_dataset_dict[input_id]
    dataset = {}

    with open(processed_intervals_file, 'r') as fp:
        data = json.load(fp)
    task_names = data['task_names']

    for dataset_id, dataset_dict in data.items():

        if dataset_id == 'task_names':
            continue

        if dataset_id not in dataset:
            dataset[dataset_id] = {'task_names': task_names}

        dataset[dataset_id]['inputs'] = all_datasets[dataset_id]

        if 'labels' in dataset_dict.keys():
            dataset[dataset_id]['labels'] = np.load(dataset_dict['labels'])

        bt = BedTool(dataset_dict['regions'])
        intervals_dict = {k: bt.to_dataframe()[k].as_matrix() for k in [
            'chrom', 'start', 'end']}
        dataset[dataset_id]['intervals'] = intervals_dict

    return dataset


def parse_inputs_and_intervals_with_holdout(processed_inputs_file, processed_intervals_file,
                                            validation_chroms=[], holdout_chroms=[]):
    dataset = parse_inputs_and_intervals(processed_inputs_file, processed_intervals_file)
    train_dataset = {}
    valid_dataset = {}

    for dataset_id, dataset_fields in dataset.items():

        intervals = dataset_fields['intervals']
        inputs = dataset_fields['inputs']
        task_names = dataset_fields['task_names']
        labels = dataset_fields.get('labels', None)

        validation_idxs = np.in1d(intervals['chrom'], validation_chroms)
        holdout_idxs = np.in1d(intervals['chrom'], holdout_chroms)
        train_idxs = ~ np.logical_or(validation_idxs, holdout_idxs)

        train_dataset[dataset_id] = {
            'task_names': task_names,
            'inputs': inputs,
        }
        train_dataset[dataset_id]['intervals'] = {k: intervals[k][train_idxs]
                                                  for k in ['chrom', 'start', 'end']}
        if labels is not None:
            train_dataset[dataset_id]['labels'] = labels[train_idxs]

        valid_dataset[dataset_id] = {
            'task_names': task_names,
            'inputs': inputs,
        }
        valid_dataset[dataset_id]['intervals'] = {k: intervals[k][validation_idxs]
                                                  for k in ['chrom', 'start', 'end']}
        if labels is not None:
            valid_dataset[dataset_id]['labels'] = labels[validation_idxs]

    return train_dataset, valid_dataset

def parse_raw_inputs(raw_inputs_file, require_consistentcy=True):
    """parses raw inputs file, returns inputs dictionary"""

    with open(raw_inputs_file, 'r') as fp:
        datasets = json.load(fp, object_pairs_hook=OrderedDict)
    # populate missing input types with None
    for dataset_id, dataset_dict in datasets.items():
        if not set(dataset_dict.keys()).issubset(RAW_INPUT_NAMES):
            err_msg = """ {} in {} has an unexpected input name!
                          Supported input names are {}""".format(
                dataset_id, raw_inputs_file, RAW_INPUT_NAMES)
            raise ValueError(err_msg)
        for input_name in RAW_INPUT_NAMES:
            if input_name not in dataset_dict:
                datasets[dataset_id][input_name] = None
    if require_consistentcy and len(datasets) > 1:
        # check which inputs are in the first dataset
        input_id2is_none = {input_id: input_val is None
                            for input_id, input_val in datasets.values()[0].items()}
        # check other datasets match it
        for dataset_id, dataset_dict in datasets.items():
            for input_id, input_val in dataset_dict.items():
                if (input_val is None) != input_id2is_none[input_id]:
                    err_msg = """ Cannot parse consistent raw input datasets:
                                  {0} in {1} inconsistent with {0}
                                  in the first dataset in {2} """.format(
                        input_id, dataset_id, raw_inputs_file)
                    raise ValueError(err_msg)

    return datasets


def parse_processed_intervals(processed_intervals_file, tasks=None):
    """Parse the processed intervals files, return a data dictionary."""
    with open(processed_intervals_file, 'r') as fp:
        data = json.load(fp, object_pairs_hook=OrderedDict)

    if tasks: # subset task_names
        task2idx = {t: i for i, t in enumerate(data['task_names'])}
        for t_name in tasks:
            if t_name not in task2idx:
                raise ValueError('Task {} was not found in intervals file {}'.format(
                    t_name, processed_intervals_file))
        task_idxs = np.array([task2idx[t] for t in tasks])
        data['task_names'] = tasks

    for dataset_id, dataset_dict in data.items():
        if dataset_id == 'task_names':
            continue
        if 'labels' in dataset_dict.keys():
            data[dataset_id]['labels'] = np.load(dataset_dict['labels'])
            if tasks:
                data[dataset_id]['labels'] = data[
                    dataset_id]['labels'][:, task_idxs]

    return data
