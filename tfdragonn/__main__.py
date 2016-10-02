from __future__ import absolute_import

import argparse
import json
import logging
import numpy as np
import os
from pybedtools import BedTool

from genomedatalayer.extractors import FastaExtractor, MemmappedFastaExtractor

from .intervals import get_tf_predictive_setup, train_test_chr_split 

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for DragoNN modeling of TF Binding.')
    # define parsers with common arguments
    data_config_parser = argparse.ArgumentParser(add_help=False)
    data_config_parser.add_argument('--data-config-file', type=str, required=True,
                                   help='file with dataset names, task names, and data files')
    model_files_parser = argparse.ArgumentParser(add_help=False)
    model_files_parser.add_argument('--arch-file', type=str, required=True,
                                    help='model architecture json file')
    model_files_parser.add_argument('--weights-file', type=str, required=True,
                                    help='weights hd5 file')
    multiprocessing_parser = argparse.ArgumentParser(add_help=False)
    multiprocessing_parser.add_argument('--n-jobs', type=int, default=1,
                                        help='num of processes used for preprocessing and postprocessing')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    # define commands 
    subparsers = parser.add_subparsers(help='tf-dragonn command help', dest='command')
    train_parser = subparsers.add_parser('train',
                                         parents=[data_config_parser,
                                                  multiprocessing_parser,
                                                  prefix_parser],
                                         help='model training help')
    interpret_parser = subparsers.add_parser('interpret',
                                             parents=[data_config_parser,
                                                      model_files_parser,
                                                      multiprocessing_parser,
                                                      prefix_parser],
                                             help='interpretation help')
    test_parser = subparsers.add_parser('test',
                                        parents=[data_config_parser,
                                                 model_files_parser,
                                                 multiprocessing_parser,
                                                 prefix_parser],
                                        help='model testing help')
    # return command and command arguments
    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def parse_data_config_file(data_config_file):
    """
    Parses data config file and returns region beds, feature beds, and data files.
    """
    data = json.load(open(data_config_file))
    if 'region_bed' not in data:
        data['region_bed'] = None
    if 'genome_data_dir' not in data:
        data['genome_data_dir'] = None
    return data


def main_train(data_config_file=None,
               prefix=None,
               n_jobs=None,
               arch_file=None,
               weights_file=None):
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    data = parse_data_config_file(data_config_file)
    # get regions and labels, split into train/test
    if os.path.isfile("{}.intervals.bed".format(prefix)) and os.path.isfile("{}.labels.npy".format(prefix)):
        logger.info("loading intervals from {}.intervals.bed".format(prefix))
        regions = BedTool("{}.intervals.bed".format(prefix))
        logger.info("loading labels from {}.labels.npy".format(prefix))
        labels = np.load("{}.labels.npy".format(prefix))
    else:
        logger.info("getting regions and labels")
        regions, labels = get_tf_predictive_setup(data['feature_beds'], region_bedtool=data['region_bed'],
                                                  bin_size=200, flank_size=400, stride=200,
                                                  filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                  save_to_prefix=prefix)
    intervals_train, intervals_test, y_train, y_test = train_test_chr_split(regions, labels, ["chr1", "chr2"])
    interval_length = intervals_train[0].length
    num_tasks = y_train.shape[1]
    architecture_parameters = {'num_filters': (35, 35, 35),
                               'conv_width': (20, 20, 20),
                               'dropout': 0.1}
    if data['genome_data_dir'] is not None: # use a streaming model
        fasta_extractor = MemmappedFastaExtractor(data['genome_data_dir'])
        logger.info("Initializing a StreamingSequenceClassifer")
        model = StreamingSequenceClassifier(interval_length, num_tasks, **architecture_parameters)
        logger.info("Compiling StreamingSequenceClassifer..")
        model.compile(y=y_train)
        logger.info("Starting to train with streaming data..")
        model.train(intervals_train, y_train, intervals_test, y_test, fasta_extractor,
                    save_best_model_to_prefix=prefix)
    else: # extract encoded data in memory
        logger.info("extracting data in memory")
        fasta_extractor = FastaExtractor(genome_fasta)
        logger.info("extracting test data")
        X_test = fasta_extractor(intervals_test)
        logger.info("extracting training data")
        X_train = fasta_extractor(intervals_train)
        # initialize model and train
        ## TODO: load pretrained model and/or architecture
        logger.info("Initializing a SequenceClassifer..")
        model = SequenceClassifier(interval_length, num_tasks, **architecture_parameters)
        logger.info("Compiling SequenceClassifer..")
        model.compile(y=y_train)
        logger.info("Starting to train")
        model.train(X_train, y_train, (X_test, y_test), patience=6)
    model.save(prefix)
    logger.info("Saved trained model files to {}.arch.json and {}.weights.h5".format(prefix, prefix))
    logger.info("Done!")


def main_interpret(data_config_file=None,
                   prefix=None,
                   n_jobs=None,
                   arch_file=None,
                   weights_file=None):
    from dlutils import write_deeplift_track
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    data = parse_data_config_file(data_config_file)
    # get regions and labels, split into train/test
    if os.path.isfile("{}.intervals.bed".format(prefix)) and os.path.isfile("{}.labels.npy".format(prefix)):
        logger.info("loading intervals from {}.intervals.bed".format(prefix))
        regions = BedTool("{}.intervals.bed".format(prefix))
        logger.info("loading labels from {}.labels.npy".format(prefix))
        labels = np.load("{}.labels.npy".format(prefix))
    else:
        logger.info("getting regions and labels")
        regions, labels = get_tf_predictive_setup(data['feature_beds'], region_bedtool=data['region_bed'],
                                                  bin_size=200, flank_size=400, stride=200,
                                                  filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                  save_to_prefix=prefix)
    if data['genome_data_dir'] is not None: # use a streaming model                                                                                                                  
        fasta_extractor = MemmappedFastaExtractor(data['genome_data_dir'])
        logger.info("Initializing a StreamingSequenceClassifer")
    else:
        logger.info("loading model...")
        model = SequenceClassifier(arch_fname=arch_file,
                                   weights_fname=weights_file)
        # extract encoded data in memory
        logger.info("extracting data from regions..")
        fasta_extractor = FastaExtractor(genome_fasta)
        X = fasta_extractor(regions)
        logger.info("running deeplift..")
        dl_scores = model.deeplift(X, batch_size=1000)
    logger.info("starting to write deeplift scores")
    for i, task_dl_scores in enumerate(dl_scores):
        logger.info("writing deeplift score track for task {}".format(str(i)))
        write_deeplift_track(task_dl_scores, regions, "{}.task{}_dl_track".format(prefix, str(i)),
                             merge_type='max',
                             CHROM_SIZES='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes')
    logger.info("Done!")


def main_test(data_config_file=None,
              prefix=None,
              n_jobs=None,
              arch_file=None,
              weights_file=None):
    from dlutils import write_deeplift_track
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    data = parse_data_config_file(data_config_file)
    # get regions and labels
    if os.path.isfile("{}.intervals.bed".format(prefix)) and os.path.isfile("{}.labels.npy".format(prefix)):
        logger.info("loading intervals from {}.intervals.bed".format(prefix))
        regions = BedTool("{}.intervals.bed".format(prefix))
        logger.info("loading labels from {}.labels.npy".format(prefix))
        labels = np.load("{}.labels.npy".format(prefix))
    else:
        logger.info("getting regions and labels")
        regions, labels = get_tf_predictive_setup(feature_beds, region_bedtool=region_bed,
                                                  bin_size=200, flank_size=400, stride=200,
                                                  filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                  save_to_prefix=prefix)
    # get test subset
    ## TODO: user-specified chromosomes
    _, intervals_test, _, y_test = train_test_chr_split(regions, labels, ["chr1", "chr2"])
    if data['genome_data_dir'] is not None: # use a streaming model
        fasta_extractor = MemmappedFastaExtractor(data['genome_data_dir'])
        logger.info("loading a StreamingSequenceClassifier model...")
        model = StreamingSequenceClassifier(arch_fname=arch_file,
                                   weights_fname=weights_file)
        logger.info("Testing the model...")
        test_results = model.test(intervals_test, y_test, fasta_extractor)
    else: # extract encoded data in memory
        logger.info("loading a SequenceClassifier model...")
        model = SequenceClassifier(arch_fname=arch_file,
                                   weights_fname=weights_file)
        # extract encoded data in memory
        logger.info("extracting data from regions..")
        fasta_extractor = FastaExtractor(genome_fasta)
        logger.info("extracting data in memory")
        fasta_extractor = FastaExtractor(genome_fasta)
        logger.info("extracting test data")
        X_test = fasta_extractor(intervals_test)
        logger.info("Testing the model...")
        test_results = model.test(X_test, y_test)
    logger.info("Test results:\n{}".format(test_results))


def main():
    # parse args
    command_functions = {'train': main_train,
                         'interpret': main_interpret,
                         'test': main_test}
    command, args = parse_args()
    # perform theano/keras import
    global SequenceClassifier, StreamingSequenceClassifier
    from .models import SequenceClassifier, StreamingSequenceClassifier
    # run command
    command_functions[command](**args)
