# tf-dragonn
Reimplementing tf binding with simpler, faster, and more stable code.

# Usage
The `tfdragonn` package provides a command-line interface with command to process data, train/test model, and interpret data using trained models. To get an overview of the interface run:
```
tfdragonn --help
```
The available commands are `memmap`, `label_regions`, `train`, `predict`, `evaluate`, `test`, and `interpret`. These commands are designed to simplify common steps in model development: data preprocessing, model training/evaluation, and prediction. In the following sections I show how to use these commands to produce competitive predictions of MYC binding for the DREAM challenge.

## Encoding raw input data
The first step is to encode raw input data into arrays that can be indexed directly during training. Run the following command to encode the hg19 genome fasta and dnase bigwigs used in the challenge:
```
tfdragonn memmap --data-config-file examples/genome_fasta_and_DNASE_fc_bigwigs.json --memmap-dir /mnt/data/memmap/TF_challenge_DNASE/ --output-file examples/genome_and_DNASE_fc_memmaped.json
```
The input data config file `examples/genome_fasta_and_DNASE_fc_bigwigs.json` has a dictionary where the keys are dataset names (in this case the name of the celltype) and the value are `genome_fasta` and `dnase_bigwig` data for that dataset. `tfdragonn memmap` creates a directory for each fasta and bigwig file in `/mnt/data/memmap/TF_challenge_DNASE/`, for example `/mnt/data/memmap/TF_challenge_DNASE/DNASE.A549.fc.signal.bigwig/` for `DNASE.A549.fc.signal.bigwig`. In each fasta/bigwig directory, there is a `.npy` file for each chromosome that holds the encoded data for that chromosome. For example `/mnt/data/memmap/TF_challenge_DNASE/DNASE.A549.fc.signal.bigwig/chr10.npy` is an array with shape `(135534747,)` that has the dnase signal value for each position in that chromosome. Similarly, `/mnt/data/memmap/TF_challenge_DNASE/hg19.genome.fa/chr10.npy` is an	array with shape `(4, 135534747)' that has the one hot encoding of the chromosome's sequence. Using these arrays, we can obtain data for any genomic interval based on its chromsome, start and end coordinates.

The output data config file `examples/genome_and_DNASE_fc_memmaped.json` provides paths to all the data directories. It also contains other attributes of datasets that have not been specified and an empty `task_names`, which we use in the next step to obtain fixed size genomic regions and their labels.

## Processing raw peak files into fixed size genomic regions and labels
Run the following command to get 1000bp genomic regions tiling DNase peaks with stride (spacing) of 200bp and binary labels for MYC binding:
```
tfdragonn label_regions --data-config-file examples/myc_peaks_on_dnase_conservative_and_memmaped_inputs.json --bin-size 200 --flank-size 400 --stride 200 --prefix /mnt/lab_data/kundaje/jisraeli/projects/TF_Challenge/models/tfdragonn_regions_and_labels/myc_new_regions_and_labels_w_ambiguous_stride200_flank400 --output-file examples/myc_conservative_dnase_regions_and_labels_stride200_flank400.json 
```
The input data config file `examples/myc_peaks_on_dnase_conservative_and_memmaped_inputs.json` has the `dnase_data_dir` and `genome_data_dir` files from the previous step for each dataset with data for MYC. For each dataset, `region_bed` points to the conservative DNase peaks in that celltype - these sepcify the subset of the genome that will be used for model training. Each DNase peak in this example is processed into bins of size 200, specified by `--bin-size`, with consecutive bins placed 200bp apart, which is specified by `--stride`. `feature_beds`, which is required for this step, points to the confident TF peaks for each TF, in this example for MYC only. `ambiguous_feature_beds`, which is optional, points to less confident MYC peaks that we want to ignore during training and evaluation. If a bin overlaps a confident peak, its labeled as positive (value of 1); if it doesn't overlap a confident peak but does overlap an ambiguous peak its labeled as ambiguous (value of -1); if it doesn't overlap any kind of peak its labeled as negative (value of 0). The labels are stored in an `npy` file whose name is based on `--prefix`, the full filename can be found in the output data config file, specified by `--output-file`, in the `labels` attribute of each dataset.

After a bin is labeled, we add extend it 400bp in each direction, specified by `--flank-size`, and the resulting fixed size regions are stored in a `.bed` file for each dataset whose name is based on `--prefix`. The full filename can be found in the output data config file in the `regions` attribute of each dataset. Besides datasets, the input and output data config files have `task_names` which is a list of all the tasks included in `feature_beds` across all datasets in the input data config file. The columns in the `labels` array in the output config file are ordered based on `task_names`. The dictionary-based specification of `feature_beds` allows for simple processing of datasets where you have a lot of tasks across all datasets but only a small subset of tasks with available data in a given dataset (in most celltypes there is data for a small fraction of total TFs assayed). 

## Model training

## Obtaining regions and corresponding predictions with trained models

## Evaluating predicted regions on a set of labeled regions

## Model testing

## Interpreting data with trained models
