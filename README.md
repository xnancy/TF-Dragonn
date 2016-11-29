# tf-dragonn
A package with command-line interface to develop, evaluate, and use production-level models of TF binding based on DragoNNs. 

# Usage
The `tfdragonn` package provides a command-line interface with command to process data, train/test model, and interpret data using trained models. To get an overview of the interface run:
```
usage: tfdragonn [-h]
                 {memmap,label_regions,train,interpret,test,predict,evaluate}
                 ...

main script for DragoNN modeling of TF Binding.

positional arguments:
  {memmap,label_regions,train,interpret,test,predict,evaluate}
                        tf-dragonn command help
    memmap              This command memory maps raw inputs inthe data config
                        file for use with streaming models,and writes a new
                        data config with memmaped inputs.
    label_regions       Generates fixed length regions and their labels for
                        each dataset.Writes a new data config file with
                        regions and labels files.
    train               model training help
    interpret           interpretation help
    test                model testing help
    predict             model predictions help
    evaluate            Predictions evaluation help

optional arguments:
  -h, --help            show this help message and exit
```
The interface is designed to simplify the standard workflow for production-level modeling of TF binding, including [processing input data](#encoding-raw-input-data), [processing output data](#Processing-raw-peak-files-into-fixed-size-genomic-regions-and-labels), [model training](#model-training), [standardizing predictions](#obtaining-regions-and-corresponding-predictions-with-trained-models), and [large scale evaluation](#evaluating-predictions-on-dnase-regions-chromosome-wide). The following sections show how to implement this workflow with `tfdragonn` to produce competitive predictions of MYC binding for the DREAM challenge.

## Encoding raw input data
The first step is to encode raw input data into arrays that can be indexed directly during training. Run the following command to encode the hg19 genome fasta and dnase bigwigs used in the challenge:
```
tfdragonn memmap --raw-inputs-config-file examples/memmap/genome_fasta_and_DNASE_fc_bigwigs.json --memmap-dir /mnt/data/memmap/TF_challenge_DNASE/ --processed-inputs-config-file examples/memmap/genome_and_DNASE_fc_memmaped.json
```
#### --raw-inputs-config-file
The raw input data config file `examples/genome_fasta_and_DNASE_fc_bigwigs.json` has a dictionary where the keys are dataset names (in this case the name of the celltype) and the value are `genome_fasta` and `dnase_bigwig` data for that dataset.

#### --memmap-dir
`tfdragonn memmap` creates a data directory for each fasta and bigwig file in the memmap directory `/mnt/data/memmap/TF_challenge_DNASE/`. For example, raw data in `DNASE.A549.fc.signal.bigwig` is encoded in the data directory `/mnt/data/memmap/TF_challenge_DNASE/DNASE.A549.fc.signal.bigwig/`. In each genome and dnase data directory, there is a `.npy` file for each chromosome that holds the encoded data for that chromosome. For example `/mnt/data/memmap/TF_challenge_DNASE/DNASE.A549.fc.signal.bigwig/chr10.npy` is an array with shape `(135534747,)` that has the dnase signal value for each position in that chromosome. Similarly, `/mnt/data/memmap/TF_challenge_DNASE/hg19.genome.fa/chr10.npy` is an array with shape `(4, 135534747)` that has the one hot encoding of the chromosome's sequence. Using these arrays, we can obtain data for any genomic interval based on its chromsome, start and end coordinates.

#### --processed-inputs-config-file
The processed inputs config file `examples/genome_and_DNASE_fc_memmaped.json` written by this command provides paths to all the data directories with encoded data. We use this file in subsequent steps to train, test, and predict.

## Processing raw peak files into fixed size genomic regions and labels
Run the following command to get 1000bp genomic regions tiling DNase peaks with stride (spacing) of 200bp and binary labels for MYC binding:
```
tfdragonn label_regions --raw-intervals-config-file examples/label_regions/myc_peaks_on_dnase_conservative_and_memmaped_inputs.json --bin-size 200 --flank-size 400 --stride 200 --prefix examples/label_regions/myc_conservative_dnase_regions_and_labels_stride200_flank400
```
#### --raw-intervals-config-file
The raw intervals config file `examples/myc_peaks_on_dnase_conservative_and_memmaped_inputs.json` has a `region_bed` for each dataset that points to the conservative DNase peaks in that celltype - these sepcify the subset of the genome that will be used for model training.  `feature_beds`, which is required for this step, points to the confident TF peaks for each TF, in this example for MYC only. `ambiguous_feature_beds`, which is optional, points to less confident MYC peaks that we want to ignore during training and evaluation.

#### --bin-size, --flank-size, and --stride
Each DNase peak in this example is processed into bins of size 200, specified by `--bin-size`, with consecutive bins placed 200bp apart, which is specified by `--stride`. If a bin overlaps a confident peak, its labeled as positive (value of 1); if it doesn't overlap a confident peak but does overlap an ambiguous peak its labeled as ambiguous (value of -1); if it doesn't overlap any kind of peak its labeled as negative (value of 0). After a bin is labeled, we add extend it 400bp in each direction, specified by `--flank-size`, resulting in regions of fixed size 1000bp that provide context for the label of the bin in the center.

#### --prefix and the processed-intervals-config-file
The labels are stored in an `npy` file whose name is based on `--prefix`, the full filename can be found in the processed-intervals-config-file `examples/label_regions/myc_conservative_dnase_regions_and_labels_stride200_flank400.json` specified by `<prefix>.json`. The fixed size regions are stored in a `.bed` file for each dataset whose name is based on `--prefix`. Besides `regions` and `labels` for each dataset, the processed-intervals-config-file also includes the `task_names`. This file is used in conjunction with the --processed-inputs-config-file for model training and testing.

## Model training
Run the following command to train a model on the myc data using the data config file with processed regions and labels:
```
tfdragonn train --data-config-file examples/label_regions/myc_conservative_dnase_regions_and_labels_stride200_flank400.json --prefix examples/train/myc_distrubted_batch_training
```
Based on the `--prefix`, this command writes a model architecture file to `examples/train/myc_distrubted_batch_training.arch.json` and a model weights file to `examples/train/myc_distrubted_batch_training.weights.h5` 

## Obtaining regions and corresponding predictions with trained models
Run the following command to obtain genomic regions and corresponding model predictions:
```
tfdragonn predict --data-config-file examples/predict/myc_relaxed_dnase_regions_and_labels_w_ambiguous_stride50_flank400.json --arch-file examples/train/myc_distrubted_batch_training.arch.json  --weights-file examples/train/myc_distrubted_batch_training.weights.h5 --test-chr chr9 --prefix examples/predict/relaxed_dnase_chr9 --output-file examples/predict/predictions.json --verbose --flank-size 400
```
The input data config file `examples/predict/myc_relaxed_dnase_regions_and_labels_w_ambiguous_stride50_flank400.json` points to DNase relaxed peaks processed with stride 50.

#### --flank-size
 `--flank-size` is a required argument that is used to trim the input genomic regions to obtain the actual core bin of each region whose label we are predicting - **make sure this corresponds to the `flank-size` used in `label_regions`, otherwise you will get bad evaluation results in the next step!**

#### --test-chr and --verbose
`--verbose` is an optional argument that, if specified, will show a progress bar. `test-chr` is another optional argument, in this case it runs predictions only for regions in chr9.

#### output-file
The output data config file `examples/predict/predictions.json` points to the `.bed` files with trimmed regions and `.npy` files with predictions for each dataset.

## Evaluating predictions on dnase regions chromosome-wide
Most TF binding sites are in DNase peaks but not all. To evaluate the performance of predictions on DNase regions in chr9 in the previous step on the entire chromosome, we first process a black list filtered chr9 into bins spanning the entire chromosome:
```
tfdragonn label_regions --data-config-file examples/evaluate/myc_peaks_on_chr9_blacklistfiltered_and_memmaped_inputs.json --bin-size 200 --flank-size 0 --stride 50 --output-file examples/evaluate/myc_chr9_blacklistfiltered_regions_and_labels_stride50_flank0.json --prefix examples/evaluate/myc_chr9_blacklistfiltered_regions_and_labels_stride50_flank0
```
Then, we evaluate predictions on DNase bins wrt chromosome-wide bins by running:
```
tfdragonn evaluate --data-config-file examples/evaluate/myc_chr9_blacklistfiltered_regions_and_labels_stride50_flank0.json --predictions-config-file examples/predict/predictions.json --stride 50
```
The evaluation is performed by "copying" the predictions on the DNase bins to the corresponding bins throughout chromosome 9 and setting predictions elsewhere to 0s. This prediction and evaluation approach maintains the FDR thersholds and effectively "corrects" the recalls to account for TF sites outside DNase regions. As we expand training to cover larger subsets of the genome, beyond DNase regions, the recalls during model training/testing will get closer to the the recalls that would come out from this evaluation.

#### --stride
`stride` is a required argument and has to match the stride used during processing of the data to perform the "copying" correctly - this operation is based on overlap between predicted regions and evaluation regions and the fraction overlap used depends on the stride.

## Formatting model predictions for the DREAM challenge
We start by running MYC predictions on test chromosomes 1, 8 and 21 in HepG2:
```
tfdragonn predict --data-config-file examples/dream_challenge/HepG2_relaxed_dnase_peaks.json --arch-file examples/train/myc_distrubted_batch_training.arch.json --weights-file examples/train/myc_distrubted_batch_training.weights.h5 --output-file examples/dream_challenge/myc_predictions_on_HepG2_relaxed_dnase_peaks.json --prefix examples/dream_challenge/myc_predictions_on_HepG2_relaxed_dnase_peaks --flank-size 400 --bin-size 200 --stride 50 --verbose --test-chr chr1 chr8 chr21
```
`bin-size` and `flank-size` are required in this call to `predict` because the data config file has a `region_bed` instead of `regions`, which means that it has to be processed on the fly to obtain predictions.

Next, we map these predictions into the DREAM format of chromosome-wide predictions by running:
```
tfdragonn map_predictions --predictions-config-file examples/dream_challenge/myc_predictions_on_HepG2_relaxed_dnase_peaks.json --target-regions examples/dream_challenge/ladder_regions.blacklistfiltered.bed.gz --stride 50 --prefix examples/dream_challenge/
```
Mapping of predictions follows the same approach as in `tfdragonn evaluate`, where target regions outside the predicted regions get probabilities of 0. The output file from this command `examples/dream_challenge/L.MYC.HepG2.tab` can be gzipped and submitted directly to the DREAM challenge.