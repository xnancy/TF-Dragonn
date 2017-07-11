from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json

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

    inputs_files = {}
    with open(processed_inputs_file, 'r') as fp:
        data = json.load(fp)
    for cur_dataset_id, cur_dataset_dict in data.items():
        inputs_files[cur_dataset_id] = {}
        for input_id in PROCESSED_INPUT_NAMES:
            if input_id in cur_dataset_dict:
                inputs_files[cur_dataset_id][
                    input_id] = cur_dataset_dict[input_id]
    dataset = {}

    with open(processed_intervals_file, 'r') as fp:
        data = json.load(fp)
    task_names = data['task_names']

    for cur_dataset_id, cur_dataset_dict in data.items():

        if cur_dataset_id == 'task_names':
            continue
        dataset[cur_dataset_id] = {
            'task_names': task_names,
            'inputs': inputs_files[cur_dataset_id],
            'intervals_file': cur_dataset_dict['intervals_file'],
        }

    return dataset


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
