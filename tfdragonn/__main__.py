from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import numpy as np
import os
from pybedtools import BedTool
from sklearn.utils import shuffle

from genomedatalayer.extractors import (
    FastaExtractor, MemmappedBigwigExtractor, MemmappedFastaExtractor
)

from .datasets import (
    parse_data_config_file, parse_raw_inputs_config_file,
    parse_processed_inputs_config_file,
    raw_input2processed_input, parse_raw_intervals_config_file,
    parse_processed_intervals_config_file
)
from .intervals import get_tf_predictive_setup, train_test_chr_split, train_valid_test_chr_split

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

input2memmap_extractor = {'dnase_bigwig': MemmappedBigwigExtractor,
                          'genome_fasta': MemmappedFastaExtractor}
processed_input2extractor = {'dnase_data_dir': MemmappedBigwigExtractor,
                             'genome_data_dir': MemmappedFastaExtractor}

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for DragoNN modeling of TF Binding.')
    # define parsers with common arguments
    data_config_parser = argparse.ArgumentParser(add_help=False)
    data_config_parser.add_argument('--data-config-file', type=str, required=True,
                                   help='file with dataset names, task names, and data files')
    raw_intervals_config_parser = argparse.ArgumentParser(add_help=False)
    raw_intervals_config_parser.add_argument('--raw-intervals-config-file', type=str, required=True,
                                             help='includes task names and map from dataset ids to raw interval files')
    processed_intervals_config_parser = argparse.ArgumentParser(add_help=False)
    processed_intervals_config_parser.add_argument('--processed-intervals-config-file', type=str, required=True,
                                                   help='includes task names and map from dataset ids to raw interval files')
    model_files_parser = argparse.ArgumentParser(add_help=False)
    model_files_parser.add_argument('--arch-file', type=str, required=True,
                                    help='model architecture json file')
    model_files_parser.add_argument('--weights-file', type=str, required=True,
                                    help='weights hd5 file')
    multiprocessing_parser = argparse.ArgumentParser(add_help=False)
    multiprocessing_parser.add_argument('--n-jobs', type=int, default=1,
                                        help='num of processes')
    output_file_parser = argparse.ArgumentParser(add_help=False)
    output_file_parser.add_argument('--output-file', type=str, required=True,
                                    help='output file to write to')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    predict_opts_parser = argparse.ArgumentParser(add_help=False)
    predict_opts_parser.add_argument('--test-chr', type=str, nargs="+", help="Chromosomes to subset test data.")
    predict_opts_parser.add_argument('--verbose', action='store_true', default=False, help="Shows prediction progress bar")
    # define commands
    subparsers = parser.add_subparsers(help='tf-dragonn command help', dest='command')
    memmap_parser = subparsers.add_parser('memmap',
                                          help="""This command memory maps raw inputs in
                                          the data config file for use with streaming models,
                                          and writes a new data config with memmaped inputs.""")
    memmap_parser.add_argument('--memmap-dir', type=str, required=True,
                               help='directories with memmaped data are created in this directory.')
    memmap_parser.add_argument('--raw-inputs-config-file', type=str, required=True, help="Raw input configuration file.")
    memmap_parser.add_argument('--processed-inputs-config-file', type=str, required=True, help="Processed input configuration file to write.")
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
    label_regions_parser.add_argument('--stride', type=int, default=200,
                                       help='spacing between consecutive bins')
    train_parser = subparsers.add_parser('train',
                                         parents=[processed_intervals_config_parser,
                                                  prefix_parser],
                                         help='model training help')
    train_parser.add_argument('--processed-inputs-config-file', type=str, required=True, help="Processed input configuration file.")
    train_parser.add_argument('--model-type', type=str, required=True, help="Name of model class. Common model types: SequenceAndDnaseClassifier, SequenceClassifier")
    train_parser.add_argument('--arch-file', type=str, required=False,
                                    help='model architecture json file')
    train_parser.add_argument('--weights-file', type=str, required=False,
                                    help='weights hd5 file')
    train_parser.add_argument('--training-config-file', type=str, required=False,
                              help='json specifying arguments for training.')
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
                                                 predict_opts_parser],
                                        help='model testing help')
    predict_parser = subparsers.add_parser('predict',
                                        parents=[data_config_parser,
                                                 model_files_parser,
                                                 output_file_parser,
                                                 predict_opts_parser,
                                                 prefix_parser],
                                        help='model predictions help')
    predict_parser.add_argument('--flank-size', type=int, required=True,
                                help='size of flanks in regions. These will be trimmed to output prediction bins.')
    predict_parser.add_argument('--bin-size', type=int, default=None, required=False,
                                      help='size of bins for processing region beds')
    predict_parser.add_argument('--stride', type=int, default=None, required=False,
                                help='spacing between consecutive bins for processing region beds')
    map_predictions_parser = subparsers.add_parser('map_predictions',
                                                   parents=[prefix_parser],
                                                   help='Mapping predictions across datasets to a uniform set of target regions')
    map_predictions_parser.add_argument('--predictions-config-file', type=str, required=True,
                                        help="Config file with regions and predictions")
    map_predictions_parser.add_argument('--target-regions', type=str, required=True,
                                        help="Predictions will be mapped to these regions")
    map_predictions_parser.add_argument('--stride', type=int, default=None, required=False,
                                        help='Spacing between regions (predicted and/or labeled). Used to determine minimum fraction overlap for scoring. Default: 50.')
    evaluate_parser = subparsers.add_parser('evaluate',
                                            parents=[data_config_parser],
                                            help='Predictions evaluation help')
    evaluate_parser.add_argument('--predictions-config-file', type=str, required=True,
                                 help="Config file with regions and predictions")
    evaluate_parser.add_argument('--stride', type=int, default=None,
                                 help='Spacing between regions (predicted and/or labeled). Used to determine minimum fraction overlap for scoring if specified. Default fraction overlap is 0.5 (assumes stride = region size).')
    evaluate_parser.add_argument('--test-chr', type=str, nargs="+", help="Chromosomes to subset data.")
    # return command and command arguments
    args = vars(parser.parse_args())
    command = args.pop("command", None)

    return command, args


def main_memmap(raw_inputs_config_file=None,
                memmap_dir=None,
                processed_inputs_config_file=None):
    """
    Memmaps every raw input in the data config file.
    Returns new data config file with memmaped inputs.
    """
    import ntpath
    raw_inputs_config = parse_raw_inputs_config_file(raw_inputs_config_file)
    dataset_id2processed_inputs = collections.OrderedDict()
    logger.info("Processing input data in {}...".format(raw_inputs_config_file))
    for dataset_id, raw_inputs in raw_inputs_config:
        processed_inputs = {}
        for input_key, raw_input_fname in raw_inputs.__dict__.items():
            if raw_input_fname is None:
                processed_inputs[raw_input2processed_input[input_key]] = None
                continue
            input_memmap_dir = os.path.join(memmap_dir, ntpath.basename(raw_input_fname))
            logger.info("Encoding {} in {}...".format(raw_input_fname, input_memmap_dir))
            extractor = input2memmap_extractor[input_key]
            try:
                extractor.setup_mmap_arrays(raw_input_fname, input_memmap_dir)
            except OSError:
                pass
            processed_inputs[raw_input2processed_input[input_key]] = input_memmap_dir
        dataset_id2processed_inputs[dataset_id] = processed_inputs
    # write json with memmaped data
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
    processed_intervals_dict = {"task_names": raw_intervals_config.task_names}
    logger.info("Generating regions and labels for datasets in {}...".format(raw_intervals_config_file))
    for dataset_id, raw_intervals in raw_intervals_config:
        logger.info("Generating regions and labels for dataset {}...".format(dataset_id))
        dataset_prefix = "{}.{}".format(prefix, dataset_id)
        path_to_dataset_regions = os.path.abspath("{}.intervals.bed".format(dataset_prefix))
        path_to_dataset_labels = os.path.abspath("{}.labels.npy".format(dataset_prefix))
        if os.path.isfile("{}.intervals.bed".format(dataset_prefix)) and os.path.isfile("{}.labels.npy".format(dataset_prefix)):
            logger.info("Regions file {} and labels file {} already exist. skipping dataset {1}!".format(path_to_dataset_regions, path_to_dataset_labels, dataset_id))
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


def main_train(processed_intervals_config_file=None,
               processed_inputs_config_file=None,
               model_type=None,
               prefix=None,
               training_config_file=None,
               arch_file=None,
               weights_file=None):
    # architecture args, need to provide json specification option
    seq_architecture_parameters = {'num_filters': (55, 75, 95),
                                   'conv_width': (20, 20, 20),
                                   'dropout': 0.1}
    logger.info("parsing inputs and intervals config files..")
    processed_intervals_config = parse_processed_intervals_config_file(processed_intervals_config_file)
    processed_inputs_config = parse_processed_inputs_config_file(processed_inputs_config_file)
    # check that inputs config has dataset ids in intervals config
    for dataset_id, _ in processed_intervals_config:
        if dataset_id not in processed_inputs_config.dataset_ids:
            raise ValueError("dataset {} in processed intervals config is not present in the processed inputs config!".format(dataset_id))
    if not processed_intervals_config.has_labels:
        raise RuntimeError("Processed intervals config file doesn't include labels for training!")
    dataset_id2regions_and_labels = {dataset_id: (BedTool(dataset.regions), np.load(dataset.labels))
                                  for dataset_id, dataset in processed_intervals_config}
    logger.info("Splitting regions and labels into train and validation subsets and shuffling train subset...")
    dataset_id2train_regions_and_labels = collections.OrderedDict()
    dataset_id2test_regions_and_labels = collections.OrderedDict()
    for dataset_id, (regions, labels) in dataset_id2regions_and_labels.items():
        ( regions_train, regions_valid, regions_test,
          y_train, y_valid, y_test ) = train_valid_test_chr_split(regions, labels, ["chr9"], ["chr1", "chr21", "chr8"])
        regions_train, y_train = shuffle(regions_train, y_train, random_state=0)
        dataset_id2train_regions_and_labels[dataset_id] = (regions_train, y_train)
        dataset_id2test_regions_and_labels[dataset_id] = (regions_valid, y_valid)
    # set up the architecture
    interval_length = dataset_id2train_regions_and_labels.values()[0][0][0].length
    num_tasks = dataset_id2train_regions_and_labels.values()[0][1].shape[1]
    # initialize model
    model_class = getattr(models, model_type)
    if arch_file is not None and weights_file is not None:
        model = model_class(arch_fname=arch_file, weights_fname=weights_file)
    else:
        model = model_class(interval_length, num_tasks) ## TODO: support user-provided constructor json
    # check if inputs config has the inputs necessary for the model
    for input_mode in model.input_modes:
        if not getattr(processed_inputs_config, "has_{}".format(input_mode)):
            raise ValueError("{} expects {} but it's not present in {}".format(model_type, input_mode, processed_inputs_config_file))
    # initialize extractors for all inputs
    dataset_id2extractors = {dataset_id: [] for dataset_id, _ in processed_intervals_config}
    for input_mode in model.input_modes:
        input_mode2extractor = {}
        for dataset_id in processed_intervals_config.dataset_ids:
            processed_inputs = processed_inputs_config.to_dict()[dataset_id]
            dataset_input_mode = processed_inputs[input_mode]
            if dataset_input_mode not in input_mode2extractor:
                extractor_type = processed_input2extractor[input_mode]
                input_mode2extractor[dataset_input_mode] = extractor_type(dataset_input_mode)
            dataset_id2extractors[dataset_id].append(input_mode2extractor[dataset_input_mode])
        logger.info("Initialized extractors for {}".format(input_mode))
    if training_config_file is not None:
        logger.info("Compiling {} with training configs in {}..".format(model_class, training_config_file))
        training_args = json.load(open(training_config_file))
        model.compile(**training_args)
    else:
        logger.info("Compiling {}..".format(model_class))
        model.compile()
    logger.info("Starting to train with streaming data..")
    model.train_on_multiple_datasets(
        dataset_id2train_regions_and_labels, dataset_id2test_regions_and_labels, dataset_id2extractors,
        task_names=processed_intervals_config.task_names, save_best_model_to_prefix=prefix)
    model.save(prefix)
    logger.info("Saved trained model files to {}.arch.json and {}.weights.h5".format(prefix, prefix))
    logger.info("Done!")


def main_test(data_config_file=None,
              test_chr=None,
              verbose=None,
              n_jobs=None,
              arch_file=None,
              weights_file=None):
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    datasets = parse_data_config_file(data_config_file)
    # get regions and labels
    if datasets.include_regions and datasets.include_labels:
        dataset2regions_and_labels = {dataset_id: (BedTool(dataset.regions), np.load(dataset.labels))
                                      for dataset_id, dataset in datasets}
    else: ## TODO: call main_label_regions, continue with output data config file
        raise RuntimeError("data config file doesnt include regions and labels for each dataset. Run the label_regions command first!")
    # get test subset if specified
    if test_chr is not None:
        logger.info("Subsetting test data to {}...".format(test_chr))
        for dataset_id, (regions, labels) in dataset2regions_and_labels.items():
            _, regions_test, _, y_test = train_test_chr_split(regions, labels, test_chr)
            dataset2regions_and_labels[dataset_id] = (regions_test, y_test)
    # get data extractors
    interval_length = dataset2regions_and_labels.values()[0][0][0].length
    dataset2extractors = datasets.get_dataset2extractors(int(interval_length/2))
    # initialize model and test
    if datasets.memmaped:
        if datasets.memmaped_fasta:
            if datasets.memmaped_dnase:
                model_class = StreamingSequenceAndDnaseClassifier
            else:
                model_class = StreamingSequenceClassifier
        logger.info("Initializing a {}".format(model_class))
        model = model_class(arch_fname=arch_file, weights_fname=weights_file)
        logger.info("Testing the model...")
        ( dataset2classification_result,
          combined_classification_result ) = model.test_on_multiple_datasets(dataset2regions_and_labels, dataset2extractors,
                                                                             batch_size=500, task_names=datasets.task_names,
                                                                             verbose=verbose)
        for dataset_id, classification_result in dataset2classification_result.items():
            print('Dataset {}:\n{}\n'.format(dataset_id, classification_result), end='')
        if len(dataset2classification_result) > 1:
            print('Metrics across all datasets:\n{}\n'.format(combined_classification_result), end='')
    else:
        raise RuntimeError("Model testing doesnt support non streaming models!")


def main_predict(data_config_file=None,
                 test_chr=None,
                 flank_size=None,
                 bin_size=None,
                 stride=None,
                 genome='hg19',
                 verbose=None,
                 n_jobs=None,
                 arch_file=None,
                 weights_file=None,
                 prefix=None,
                 output_file=None):
    from .intervals import filter_interval_by_chrom, remove_flanks, bin_bed, filter_interval_by_length
    # get matched region beds and feature/tf beds
    logger.info("parsing data config file..")
    datasets = parse_data_config_file(data_config_file)
    # get regions and labels
    if datasets.include_regions:
        dataset2regions = {dataset_id: BedTool(dataset.regions) for dataset_id, dataset in datasets}
    else: ## TODO: call main_label_regions, continue with output data config file
        logger.info("data config file doesnt include regions and labels for each dataset. Processing data into fixed size regions..")
        if not datasets.include_regions_or_region_bed:
            raise RuntimeError("Found at least one dataset without regions that doesn't include a region bed. Exiting!")
        if bin_size is None or stride is None:
            raise RuntimeError("--bin-size and --stride must be specified to process data!")
        data_dict = datasets.to_dict()
        dataset2regions = {}
        for dataset_id, dataset in datasets:
            if dataset.regions is not None:
                dataset2regions[dataset_id] = BedTool(dataset.regions)
            else:
                logger.info("Procesing region bed in dataset {}..".format(dataset_id))
                bins = bin_bed(BedTool(dataset.region_bed), bin_size=bin_size, stride=stride)
                bins = bins.set_chromsizes(genome)
                regions = bins.slop(b=flank_size)
                regions = regions.each(filter_interval_by_length, bin_size + 2*flank_size).saveas()
                dataset2regions[dataset_id] = regions
    # get test subset if specified
    if test_chr is not None:
        logger.info("Subsetting test data to {}...".format(test_chr))
        for dataset_id, regions in dataset2regions.items():
            dataset2regions[dataset_id] = regions.each(filter_interval_by_chrom, test_chr).saveas()
    # get data extractors
    interval_length = dataset2regions.values()[0][0].length
    dataset2extractors = datasets.get_dataset2extractors(int(interval_length/2))
    # initialize model and test
    if datasets.memmaped:
        if datasets.memmaped_fasta:
            if datasets.memmaped_dnase:
                model_class = StreamingSequenceAndDnaseClassifier
            else:
                model_class = StreamingSequenceClassifier
        logger.info("Initializing a {}".format(model_class))
        model = model_class(arch_fname=arch_file, weights_fname=weights_file)
        logger.info("Running predictions...")
        data_dict = datasets.to_dict()
        for dataset_id, regions in dataset2regions.items():
            preds = model.predict(regions, dataset2extractors[dataset_id], batch_size=500, verbose=verbose)
            regions_fname = os.path.abspath("{}.{}.regions.bed".format(prefix, dataset_id))
            preds_fname = os.path.abspath("{}.{}.predictions.npy".format(prefix, dataset_id))
            # trim flanks from regions
            regions = regions.each(remove_flanks, flank_size)
            BedTool(regions).saveas().moveto(regions_fname)
            np.save(preds_fname, preds)
            for key, value in data_dict[dataset_id].items():
                if key == "regions":
                    data_dict[dataset_id][key] = regions_fname
                elif key == "labels":
                    data_dict[dataset_id][key] = preds_fname
                else:
                    data_dict[dataset_id][key] = None
            logger.info("Saved {} dataset regions to {} and predictions to {}".format(dataset_id, regions_fname, preds_fname))
        json.dump(data_dict, open(output_file, "w"), indent=4)
        logger.info("Wrote prediction data config file to {}.".format(output_file))
        logger.info("Done!")
    else:
        raise RuntimeError("tfdragonn predict doesnt support non streaming models!")

def main_evaluate(data_config_file=None,
                  predictions_config_file=None,
                  test_chr=None,
                  stride=None):
    from .intervals import bed_intersection_scores
    from .metrics import ClassificationResult
    # get regions and labels
    logger.info("parsing data config file..")
    datasets = parse_data_config_file(data_config_file)
    if datasets.include_regions and datasets.include_labels:
        dataset2regions_and_labels = {dataset_id: (BedTool(dataset.regions), np.load(dataset.labels))
                                      for dataset_id, dataset in datasets}
    else: ## TODO: call main_label_regions, continue with output data config file
        raise RuntimeError("data config file doesnt include regions and labels for each dataset. Run the label_regions command first!")
    # subset regions and labels if test-chr is specified
    # get test subset if specified
    if test_chr is not None:
        logger.info("Subsetting data to {}...".format(test_chr))
        for dataset_id, (regions, labels) in dataset2regions_and_labels.items():
            ( _, regions_test,
              _, y_test ) = train_test_chr_split(regions, labels, ["chr9"])
            dataset2regions_and_labels[dataset_id] = (regions_test, y_test)
    # get predicted regions and predicted probabilities
    logger.info("parsing predictions config file..")
    datasets_preds = parse_data_config_file(predictions_config_file)
    if datasets_preds.include_regions and datasets_preds.include_labels:
        dataset2preds_regions_and_labels = {dataset_id: (BedTool(dataset.regions), np.load(dataset.labels))
                                            for dataset_id, dataset in datasets_preds}
    # intersect predictions and get classification metrics
    logger.info("Evaluating predicted regions against all regions..")
    dataset2classification_result = {}
    predictions_list = []
    labels_list = []
    for dataset_id, (preds_regions, preds_labels) in dataset2preds_regions_and_labels.items():
        logger.info("Evaluating predictions in dataset {}...".format(dataset_id))
        preds_df = BedTool(preds_regions).to_dataframe()
        preds_df.iloc[:, -1] = preds_labels
        preds_bedtool = BedTool.from_dataframe(preds_df)
        regions, labels = dataset2regions_and_labels[dataset_id]
        if stride is not None:
            interval_length = dataset2regions_and_labels.values()[0][0][0].length
            f = 1 - (stride - 1) / interval_length
            F = 1 - (stride - 1) / interval_length
        else:
            f = 1
            F = 1
        predictions = bed_intersection_scores(BedTool(regions), preds_bedtool, score_index=4, f=f, F=F)
        dataset2classification_result[dataset_id] = ClassificationResult(labels, predictions)
        predictions_list.append(predictions)
        labels_list.append(labels)
    for dataset_id, classification_result in dataset2classification_result.items():
        print('Dataset {}:\n{}\n'.format(dataset_id, classification_result), end='')
    if len(dataset2classification_result) > 1:
        predictions = np.vstack(predictions_list)
        y = np.vstack(labels_list)
        combined_classification_result = ClassificationResult(y, predictions)
        print('Metrics across all datasets:\n{}\n'.format(combined_classification_result), end='')


def main_map_predictions(predictions_config_file=None,
                         target_regions=None,
                         prefix=None,
                         stride=None):
    from .intervals import bed_intersection_scores
    import pandas as pd
    # get predicted regions and predicted probabilities
    logger.info("parsing predictions config file and target regions..")
    datasets_preds = parse_data_config_file(predictions_config_file)
    if len(datasets_preds.task_names) > 1:
        raise RuntimeError("map_predictions doesn't support mapping of multitask predictions!")
    if datasets_preds.include_regions and datasets_preds.include_labels:
        dataset2preds_regions_and_labels = {dataset_id: (BedTool(dataset.regions), np.load(dataset.labels))
                                            for dataset_id, dataset in datasets_preds}
    else:
        raise RuntimeError("The predictions config file doesnt include predicted regions and labels for each dataset!")
    # intersect predictions and get classification metrics
    logger.info("Mapping predictions to target regions..")
    task_name = datasets_preds.task_names[0]
    for dataset_id, (preds_regions, preds_labels) in dataset2preds_regions_and_labels.items():
        logger.info("Mapping predictions in dataset {}...".format(dataset_id))
        preds_df = BedTool(preds_regions).to_dataframe()
        preds_df.iloc[:, 3] = preds_labels
        preds_bedtool = BedTool.from_dataframe(preds_df)
        if stride is not None:
            interval_length = preds_bedtool[0].length
            f = 1 - (stride - 1) / interval_length
            F = 1 - (stride - 1) / interval_length
        else:
            f = 1
            F = 1
        predictions = bed_intersection_scores(BedTool(target_regions), preds_bedtool, score_index=4, f=f, F=F)
        target_regions_df = BedTool(target_regions).to_dataframe()
        target_regions_df = target_regions_df.join(pd.DataFrame(predictions))
        target_regions_and_predictions = BedTool.from_dataframe(target_regions_df)
        target_regions_and_predictions.saveas("{}L.{}.{}.tab.gz".format(prefix, task_name, dataset_id))
        logger.info("Saved target regions with predictions to {}L.{}.{}.tab".format(prefix, task_name, dataset_id))
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


def main():
    # parse args
    command_functions = {'memmap': main_memmap,
                         'label_regions': main_label_regions,
                         'train': main_train,
                         'predict': main_predict,
                         'map_predictions': main_map_predictions,
                         'evaluate': main_evaluate,
                         'interpret': main_interpret,
                         'test': main_test}
    command, args = parse_args()
    if command in ['train', 'predict', 'interpret', 'test']: # perform theano/keras import
        global SequenceClassifier, StreamingSequenceClassifier, StreamingSequenceAndDnaseClassifier
        from .models import SequenceClassifier, StreamingSequenceClassifier, StreamingSequenceAndDnaseClassifier
        global models
        from tfdragonn import models
    # run command
    command_functions[command](**args)
