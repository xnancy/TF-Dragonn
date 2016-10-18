from __future__ import absolute_import

import argparse
import json
import logging
import numpy as np
import os
from pybedtools import BedTool
from sklearn.utils import shuffle

from genomedatalayer.extractors import (
    FastaExtractor, MemmappedBigwigExtractor, MemmappedFastaExtractor
)

from .datasets import Dataset, parse_data_config_file
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

RAW_INPUT_KEYS = ['dnase_bigwig', 'genome_fasta']
input2memmap_extractor = {'dnase_bigwig': MemmappedBigwigExtractor,
                          'genome_fasta': MemmappedFastaExtractor}
input2memmap_input = {'dnase_bigwig': 'dnase_data_dir',
                      'genome_fasta': 'genome_data_dir'}

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
    output_file_parser = argparse.ArgumentParser(add_help=False)
    output_file_parser.add_argument('--output-file', type=str, required=True,
                                    help='output file to write to')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    memmap_dir_parser = argparse.ArgumentParser(add_help=False)
    memmap_dir_parser.add_argument('--memmap-dir', type=str, required=True,
                               help='directories with memmaped data are created in this directory.')
    # define commands 
    subparsers = parser.add_subparsers(help='tf-dragonn command help', dest='command')
    memmap_parser = subparsers.add_parser('memmap',
                                         parents=[data_config_parser,
                                                  memmap_dir_parser,
                                                  output_file_parser],
                                         help='This command memory maps raw inputs in'+
                                              'the data config file for use with streaming models,'+
                                              'and writes a new data config with memmaped inputs. ')
    label_regions_parser = subparsers.add_parser('label_regions',
                                                 parents=[data_config_parser,
                                                          multiprocessing_parser,
                                                          output_file_parser,
                                                          prefix_parser],
                                         help='Generates fixed length regions and their labels for each dataset.'
                                              'Writes a new data config file with regions and labels files.')
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


def main_memmap(data_config_file=None,
                memmap_dir=None,
                output_file=None):
    """
    Memmaps every raw input in the data config file.
    Returns new data config file with memmaped inputs.
    """
    import ntpath
    data = parse_data_config_file(data_config_file)
    logger.info("Memapping input data in {}...".format(data_config_file))
    for dataset_id, dataset in data.items():
        for input_key in RAW_INPUT_KEYS:
            raw_input_fname = getattr(dataset, input_key)
            if raw_input_fname is not None:
                input_memmap_dir = os.path.join(memmap_dir, ntpath.basename(raw_input_fname))
                logger.info("Encoding {} in {}...".format(raw_input_fname, input_memmap_dir))
                extractor = input2memmap_extractor[input_key]
                extractor.setup_mmap_arrays(raw_input_fname, input_memmap_dir)
                # update the dataset in data
                memmap_dataset_dict = dataset.to_dict()
                del memmap_dataset_dict[input_key]
                memmap_dataset_dict[input2memmap_input[input_key]] = input_memmap_dir
                data[dataset_id] = memmap_dataset_dict
                logger.info("Replaced {}: {} with\n\t\t\t\t\t\t  {}: {} in\n\t\t\t\t\t\t  {} dataset".format(
                    input_key, raw_input_fname, input2memmap_input[input_key], input_memmap_dir, dataset_id))
    # write json with memmaped data
    json.dump(data, open(output_file, "w"), indent=4)
    logger.info("Wrote memaped data config file to {}.".format(output_file))
    logger.info("Done!")

def main_label_regions(data_config_file=None,
                       n_jobs=None,
                       prefix=None,
                       output_file=None):
    """
    Generates regions and labels files for each dataset.
    Writes new data config file with the generated files.
    """
    data = parse_data_config_file(data_config_file)
    data_dict = data.to_dict()
    logger.info("Generating regions and labels for datasets in {}...".format(data_config_file))
    for dataset_id, dataset in data:
        logger.info("Generating regions and labels for dataset {}...".format(dataset_id))
        dataset_prefix = "{}.{}".format(prefix, dataset_id)
        if os.path.isfile("{}.intervals.bed".format(dataset_prefix)) and os.path.isfile("{}.labels.npy".format(dataset_prefix)):
            logger.info("Regions file {0}.intervals.bed and labels file {0}.labels.npy already exists. skipping dataset {1}!".format(dataset_prefix, dataset_id))
        else:
            regions, labels = get_tf_predictive_setup(dataset.feature_beds, region_bedtool=dataset.region_bed,
                                                      bin_size=200, flank_size=400, stride=200,
                                                      filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                      save_to_prefix=dataset_prefix)
            logger.info("Saved regions to {0}.intervals.bed and labels to {0}.labels.npy".format(dataset_prefix))
        # update the data dictionary
        labeled_regions_dataset_dict = dataset.to_dict()
        del labeled_regions_dataset_dict["feature_beds"]
        del labeled_regions_dataset_dict["region_bed"]
        labeled_regions_dataset_dict["regions"] = os.path.abspath("{}.intervals.bed".format(dataset_prefix))
        labeled_regions_dataset_dict["labels"] = os.path.abspath("{}.labels.npy".format(dataset_prefix))
        data_dict[dataset_id] = labeled_regions_dataset_dict
        logger.info("Replaced bed files with regions and labels files for dataset {}.".format(dataset_id))
    # write json with regions and labels files
    json.dump(data_dict, open(output_file, "w"), indent=4)
    logger.info("Wrote new data config file to {}.".format(output_file))
    logger.info("Done!")


def main_train(data_config_file=None,
               prefix=None,
               n_jobs=None,
               arch_file=None,
               weights_file=None):
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    datasets = parse_data_config_file(data_config_file)
    # get regions and labels, split into train/test
    if len(datasets) > 1:
        raise RuntimeError("Data configurations with more than one datasets are not supported yet!")
    else:
        data = datasets[datasets.keys()[0]]
    if os.path.isfile("{}.intervals.bed".format(prefix)) and os.path.isfile("{}.labels.npy".format(prefix)):
        logger.info("loading intervals from {}.intervals.bed".format(prefix))
        regions = BedTool("{}.intervals.bed".format(prefix))
        logger.info("loading labels from {}.labels.npy".format(prefix))
        labels = np.load("{}.labels.npy".format(prefix))
    elif data.regions is not None and data.labels is not None:
        if os.path.isfile(data.regions) and os.path.isfile(data.labels):
            logger.info("loading intervals from {}".format(data.regions))
            regions = BedTool(data.regions)
            logger.info("loading labels from {}".format(data.labels))
            labels = np.load(data.labels)
    else:
        logger.info("getting regions and labels")
        regions, labels = get_tf_predictive_setup(data.feature_beds, region_bedtool=data.region_bed,
                                                  bin_size=200, flank_size=400, stride=200,
                                                  filter_flank_overlaps=False, genome='hg19', n_jobs=n_jobs,
                                                  save_to_prefix=prefix)
    intervals_train, intervals_test, y_train, y_test = train_test_chr_split(regions, labels, ["chr1", "chr2"])
    # shuffle training intervals and labels
    intervals_train, y_train = shuffle(intervals_train, y_train, random_state=0)
    interval_length = intervals_train[0].length
    num_tasks = y_train.shape[1]
    architecture_parameters = {'num_filters': (35, 35, 35),
                               'conv_width': (20, 20, 20),
                               'dropout': 0.1}
    if data.genome_data_dir is not None: # use a streaming model
        fasta_extractor = MemmappedFastaExtractor(data.genome_data_dir)
        logger.info("Initializing a StreamingSequenceClassifer")
        model = StreamingSequenceClassifier(interval_length, num_tasks, **architecture_parameters)
        logger.info("Compiling StreamingSequenceClassifer..")
        model.compile(y=y_train)
        logger.info("Starting to train with streaming data..")
        model.train(intervals_train, y_train, intervals_test, y_test, fasta_extractor,
                    save_best_model_to_prefix=prefix)
    else: # extract encoded data in memory
        logger.info("extracting data in memory")
        fasta_extractor = FastaExtractor(data.genome_fasta)
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
        model = StreamingSequenceClassifier(arch_fname=arch_file,
                                            weights_fname=weights_file)
        logger.info("running deeplift..")
        dl_scores = model.deeplift(regions, fasta_extractor, batch_size=1000)
        dl_scores = dl_scores.sum(axis=3, keepdims=True)
    else:
        logger.info("loading model...")
        model = SequenceClassifier(arch_fname=arch_file,
                                   weights_fname=weights_file)
        # extract encoded data in memory
        logger.info("extracting data from regions..")
        fasta_extractor = FastaExtractor(data['genome_fasta'])
        X = fasta_extractor(regions)
        logger.info("running deeplift..")
        dl_scores = model.deeplift(X, batch_size=1000)
        dl_scores = dl_scores.sum(axis=3, keepdims=True)
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
        logger.info("extracting data in memory")
        fasta_extractor = FastaExtractor(data['genome_fasta'])
        logger.info("extracting test data")
        X_test = fasta_extractor(intervals_test)
        logger.info("Testing the model...")
        test_results = model.test(X_test, y_test)
    logger.info("Test results:\n{}".format(test_results))


def main():
    # parse args
    command_functions = {'memmap': main_memmap,
                         'label_regions': main_label_regions,
                         'train': main_train,
                         'interpret': main_interpret,
                         'test': main_test}
    command, args = parse_args()
    if command in ['train', 'interpret', 'test']: # perform theano/keras import
        global SequenceClassifier, StreamingSequenceClassifier
        from .models import SequenceClassifier, StreamingSequenceClassifier
    # run command
    command_functions[command](**args)
