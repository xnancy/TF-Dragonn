#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import logging
import numpy as np
import pybedtools

from keras import backend as K
import tensorflow as tf

from tfdragonn.datasets import parse_raw_intervals_config_file  # TODO:

import genomeflow_interface
import models
import trainers
from intervals import get_tf_predictive_setup


DIR_PREFIX = '/srv/scratch/tfbinding/'
LOGDIR_PREFIX = '/srv/scratch/tfbinding/tf_logs/'

HOLDOUT_CHROMS = ['chr1', 'chr8', 'chr21']
VALID_CHROMS = ['chr9']

EARLYSTOPPING_KEY = 'auPRC'
EARLYSTOPPING_PATIENCE = 4
# EARLYSTOPPING_TOLERANCE = 1e-4

IN_MEMORY = False
# BATCH_SIZE = 128
BATCH_SIZE = 256
# EPOCH_SIZE = 250000
EPOCH_SIZE = 2500000
# EPOCH_SIZE = 5000000
LEARNING_RATE = 0.0003
# LEARNING_RATE=0.00003

# TF Session Settings
DEFER_DELETE_SIZE = int(250 * 1e6)  # 250MB
GPU_MEM_PROP = 0.45  # Allows 2x sessions / gpu


logging.basicConfig(
    format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('train-wrapper')

backend = K.backend()
if backend != 'tensorflow':
    raise ValueError(
        'Only the keras tensorflow backend is supported, currently using {}'.format(backend))


def parse_args():
    parser = argparse.ArgumentParser('main TF-DragoNN script')
    subparsers = parser.add_subparsers(
        help='tf-dragonn command help', dest='command')

    predict_parser = subparsers.add_parser(
        'predict', help="main prediction script")
    predict_parser.add_argument('datasetspec', type=os.path.abspath,
                                help='Dataset parameters json file path')
    predict_parser.add_argument('intervalspec', type=os.path.abspath,
                                help='Interval parameters json file path')
    predict_parser.add_argument('logdir', type=os.path.abspath,
                                help='Model log directory')
    predict_parser.add_argument('prefix', type=os.path.abspath,
                                help='prefix to bedGraphs with predictions')
    predict_parser.add_argument('--visiblegpus', type=str,
                                required=True, help='Visible GPUs string')
    predict_parser.add_argument('--flank-size', type=int, default=400,
                                help='Trims input intervals by this size. Default: 400.')

    label_regions_parser = subparsers.add_parser('label_regions', formatter_class=argparse.RawTextHelpFormatter,
                                                 help='Generates fixed length regions and their labels for each dataset.'
                                                 'Writes an intervalspec file.')
    label_regions_parser.add_argument('raw_intervals_config_file', type=str,
                                      help='includes task names and map from dataset ids to raw interval files')
    label_regions_parser.add_argument(
        'prefix', type=str, help='prefix to output files')
    label_regions_parser.add_argument('--n-jobs', type=int, default=1,
                                      help='num of processes.\nDefault: 1.')
    label_regions_parser.add_argument('--bin-size', type=int, default=200,
                                      help='size of bins for labeling.\nDefault: 200.')
    label_regions_parser.add_argument('--flank-size', type=int, default=400,
                                      help='size of flanks around labeled bins.\nDefault: 400.')
    label_regions_parser.add_argument('--stride', type=int, default=50,
                                      help='spacing between consecutive bins.\nDefault: 50.')
    label_regions_parser.add_argument('--genome', type=str, default='hg19',
                                      help='Genome name.\nDefault: hg19.'
                                      '\nOptions: hg18, hg38, mm9, mm10, dm3, dm6.')

    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def predict_tf_dragonn(datasetspec, intervalspec, logdir, visiblegpus, flank_size, prefix):
    datasetspec = os.path.abspath(datasetspec)
    assert(os.path.isfile(datasetspec))
    assert(datasetspec.startswith(DIR_PREFIX))

    intervalspec = os.path.abspath(intervalspec)
    assert(os.path.isfile(intervalspec))
    assert(intervalspec.startswith(DIR_PREFIX))

    logdir = os.path.abspath(logdir)
    assert(os.path.exists(logdir))
    assert(logdir.startswith(LOGDIR_PREFIX))

    modelspec = os.path.join(logdir, 'modelspec.json')
    assert(os.path.isfile(modelspec))

    logger.info('dataspec file: {}'.format(datasetspec))
    logger.info('intervalspec file: {}'.format(intervalspec))
    logger.info('logdir path: {}'.format(logdir))
    logger.info('visiblegpus string: {}'.format(visiblegpus))

    logger.info("Setting up keras session")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_PROP
    session = tf.Session(config=session_config)
    K.set_session(session)

    logger.info('Setting up genomeflow queues')
    data_interface = genomeflow_interface.GenomeFlowInterface(
        datasetspec, intervalspec, modelspec, validation_chroms=HOLDOUT_CHROMS, holdout_chroms=[])
    example_queues = {dataset_id: data_interface.get_example_queue(dataset_values, dataset_id,
                                                                   num_epochs=1,
                                                                   input_names=data_interface.input_names,
                                                                   enqueues_per_thread=[128, 1])
                      for dataset_id, dataset_values in data_interface.validation_dataset.items()}

    logger.info('loading  model and trainer')
    model = models.model_from_minimal_config(modelspec,
                                             example_queues.values()[
                                                 0].output_shapes,
                                             len(data_interface.task_names))
    model.load_weights(os.path.join(logdir, 'model.weights.h5'))
    trainer = trainers.ClassifierTrainer()

    def generate_intervals(chroms, starts, ends, preds):
        for chrom, start, end, pred in zip(chroms, starts, ends, preds):
            yield pybedtools.create_interval_from_list([chrom, start, end, str(pred)])

    for dataset_id, example_queue in example_queues.items():
        logger.info('generating predictions for dataset {}'.format(dataset_id))
        intervals, predictions = trainer.predict(model, example_queue)

        # trim flanks
        intervals['start'] += flank_size
        intervals['end'] -= flank_size

        # write each task to bedtool and save
        for task_indx, task_name in enumerate(data_interface.task_names):
            intervals = generate_intervals(intervals['chrom'],
                                           intervals['start'],
                                           intervals['end'],
                                           predictions[:, task_indx])
            bedtool = pybedtools.BedTool(intervals)
            output_fname = "{}.{}.{}.tab.gz".format(
                prefix, task_name, dataset_id)
            bedtool.sort().saveas(output_fname)
            logger.info("\nSaved {} predictions in dataset {} to {}".format(
                task_name, dataset_id, output_fname))
    logger.info('Done!')


def main_label_regions(raw_intervals_config_file, prefix,
                       n_jobs=1, bin_size=200, flank_size=400, stride=50, genome='hg19'):
    """
    Generates regions and labels files for each dataset.
    Writes new data config file with the generated files.
    """
    raw_intervals_config = parse_raw_intervals_config_file(
        raw_intervals_config_file)
    processed_intervals_dict = collections.OrderedDict(
        [("task_names", raw_intervals_config.task_names)])
    logger.info("Generating regions and labels for datasets in {}...".format(
        raw_intervals_config_file))
    for dataset_id, raw_intervals in raw_intervals_config:
        logger.info(
            "Generating regions and labels for dataset {}...".format(dataset_id))
        path_to_dataset_intervals_file = os.path.abspath(
            "{}.{}.intervals_file.tsv.gz".format(prefix, dataset_id))
        if os.path.isfile(path_to_dataset_intervals_file):
            logger.info("intervals_file file {} already exists. skipping dataset {}!".format(
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
            #np.save(path_to_dataset_intervals_file, intervals_file_array)
            np.savetxt(path_to_dataset_intervals_file,
                       intervals_file_array, delimiter='\t', fmt='%s')
            logger.info("Saved intervals_file file to {}".format(
                path_to_dataset_intervals_file))
        processed_intervals_dict[dataset_id] = {
            "intervals_file": path_to_dataset_intervals_file}
    # write processed intervals config file
    processed_intervals_config_file = os.path.abspath("{}.json".format(prefix))
    json.dump(processed_intervals_dict, open(
        processed_intervals_config_file, "w"), indent=4)
    logger.info("Wrote new data config file to {}.".format(
        processed_intervals_config_file))
    logger.info("Done!")


def main():
    command_functions = {'train': train_tf_dragonn,
                         'test': test_tf_dragonn,
                         'predict': predict_tf_dragonn,
                         'label_regions': main_label_regions}
    command, args = parse_args()
    command_functions[command](**args)


if __name__ == '__main__':
    main()
