from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json

import numpy as np
from pybedtools import BedTool

def parse_inputs_and_intervals(processed_inputs_file, processed_intervals_file):
    """Parse the processed inputs and intervals files, return a dataset dictionary."""

    # TODO(cprobert): modify this to only get the required datasets, and throw
    # an error if they are not present

    all_datasets = {}
    with open(processed_inputs_file, 'r') as fp:
        data = json.load(fp)
    for cur_dataset_id, cur_dataset_dict in data.items():
        all_datasets[cur_dataset_id] = {}
        if 'dnase_data_dir' in cur_dataset_dict:
            all_datasets[cur_dataset_id][
                'dnase_data_dir'] = cur_dataset_dict['dnase_data_dir']
        if 'genome_data_dir' in cur_dataset_dict:
            all_datasets[cur_dataset_id][
                'genome_data_dir'] = cur_dataset_dict['genome_data_dir']
        if 'dnase_bed_counts_data_dir' in cur_dataset_dict:
            all_datasets[cur_dataset_id][
                'dnase_peaks_counts_data_dir'] = cur_dataset_dict['dnase_peaks_counts_data_dir']

    dataset = {}

    with open(processed_intervals_file, 'r') as fp:
        data = json.load(fp)
    task_names = data['task_names']
    for dataset_id, dataset_dict in data.items():
        if dataset_id == 'task_names':
            continue
        if dataset_id not in dataset:
            dataset[dataset_id] = {}
        dataset[dataset_id]['inputs'] = all_datasets[dataset_id]

        # transpose for efficent slicing
        labels_mtx = np.load(dataset_dict['labels']).T
        labels_dict = {task_id: labels_mtx[i, :]
                       for i, task_id in enumerate(task_names)}
        dataset[dataset_id]['labels'] = labels_dict

        bt = BedTool(dataset_dict['regions'])
        regions_dict = {k: bt.to_dataframe()[k].as_matrix() for k in [
            'chrom', 'start', 'end']}
        dataset[dataset_id]['intervals'] = regions_dict

    return dataset


RAW_INPUT_NAMES = ['genome_fasta', 'dnase_bigwig', 'dnase_peaks_bed']


def parse_raw_inputs(raw_inputs_file, require_consistentcy=True):
    """parses raw inputs file, returns inputs dictionary"""

    with open(raw_inputs_file, 'r') as fp:
        datasets = json.load(fp, object_pairs_hook=OrderedDict)
    # populate missing input types with None
    for dataset_id, dataset_dict in datasets.items():
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
