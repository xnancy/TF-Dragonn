from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
from pybedtools import BedTool


def parse_inputs_and_intervals(processed_inputs_file, processed_intervals_file):
    """Parse the processed inputs and intervals files, return a dataset dictionary."""

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

        dataset[dataset_id]['labels'] = np.load(dataset_dict['labels'])

        bt = BedTool(dataset_dict['regions'])
        intervals_dict = {k: bt.to_dataframe()[k].as_matrix() for k in [
            'chrom', 'start', 'end']}
        dataset[dataset_id]['intervals'] = intervals_dict

    return dataset
