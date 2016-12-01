from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
