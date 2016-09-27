from __future__ import absolute_import

import argparse
import json
import logging

from genomedatalayer.extractors import FastaExtractor

from .intervals import get_tf_predictive_setup, train_test_chr_split 

logger = logging.getLogger('tf-dragonn')

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
    return data['region_bed'], data['feature_beds'], data['genome_fasta']


def main_train(data_config_file=None,
               prefix=None,
               n_jobs=None,
               model_file=None,
               weights_file=None):
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    region_bed, feature_beds, genome_fasta = parse_data_config_file(data_config_file)
    # get regions and labels, split into train/test
    ## TODO: save/load regions and labels
    logger.info("getting regions and labels")
    regions, labels = get_tf_predictive_setup(feature_beds, region_bedtool=region_bed,
                                              bin_size=200, flank_size=400, stride=200,
                                              filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs)
    intervals_train, intervals_test, y_train, y_test = train_test_chr_split(regions, labels, ["chr1", "chr2"])
    # extract encoded data in memory
    logger.info("extracting data in memory")
    fasta_extractor = FastaExtractor(genome_fasta)
    logger.info("extracting test data")
    X_test = fasta_extractor(intervals_test)
    logger.info("extracting training data")
    X_train = fasta_extractor(intervals_train)
    # initialize model and train
    ## TODO: load pretrained model and/or architecture
    logger.info("Initializing a SequenceClassifer")
    model = SequenceClassifier(X_train, y_train, num_filters=(35, 35, 35), conv_width=(20, 20, 20), dropout=0.1)
    logger.info("Starting to train")
    model.train(X_train, y_train, (X_test, y_test), patience=6)
    logger.info("Saving trained model")
    model.save(prefix)
    ## TODO: deeplift, browser tracks, motif disocvery 


def main():
    # parse args
    command_functions = {'train': main_train}
    command, args = parse_args()
    # perform theano/keras import
    global SequenceClassifier
    from .models import SequenceClassifier
    # run command
    command_functions[command](**args)
