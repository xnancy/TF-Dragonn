# tf-dragonn
Reimplementing tf binding with simpler, faster, and more stable code.

## Usage
The `tfdragonn` package provides a command-line interface with command to process data, train/test model, and interpret data using trained models. To get an overview of the interface run:
```
tfdragonn --help
```
The available commands are `memmap`, `label_regions`, `train`, `test`, and `interpret`. See sections below for example usage of each command.

### Memory mapping raw input data
The data config file supports `genome_fasta` and `dnase_bigwig` inputs for quick model prototyping. When these inputs are used for training/testing of models, `tfdragonn` will encode them based on the dataset intervals in into arrays that will be in memory throughout training/testing. For large scale data that cannot fit in memory, we first encode each input genome-wide into binary files that are then memory mapped to stream data for model training/testing. The `memmap` command performs genome-wide encoding of every raw input in a data config file and writes a new data config file with the binary inputs. Run this command to encode DNase foldchange bigwigs in 14 celltypes used for the TF binding challenge:
```
tfdragonn memmap --data-config-file examples/DNASE_fc_bigwigs.json --memmap-dir ./large_scale_encoding_example --output-file examples/DNASE_fc_memmaped.json
```
`examples/DNASE_fc_memmaped.json` is the new data config file with encoded inputs. Here is another example command that encodes DNase foldchange bigwigs and the human genome fasta:
```
tfdragonn memmap --data-config-file examples/genome_fasta_and_DNASE_fc_bigwigs.json --memmap-dir /mnt/data/memmap/TF_challenge_DNASE/ --output-file examples/genome_and_DNASE_fc_memmaped.json
```

### Processing raw peak files into labeled regions
The `tfdragonn label_regions` command provides a simple way to process datasets with raw peaks files into sets of regions of fixed length and the corresponding labels. `examples/TF_peaks_and_memmaped_fasta_DNASE_training.json` is an example config file with all of the reproducible TF peak data in the TF binding challenge. We process these peaks into pairs of regions and label by running:
```
tfdragonn label_regions --data-config-file examples/TF_peaks_and_memmaped_fasta_DNASE_training.json --n-jobs 16 --output-file examples/regions_and_labels_for_TF_peaks_and_memmaped_fasta_DNASE_training.json --prefix /mnt/lab_data/kundaje/jisraeli/projects/TF_Challenge/models/tfdragonn_regions_and_labels/TF_peaks
```
The new data config file, `examples/regions_and_labels_for_TF_peaks_and_memmaped_fasta_DNASE_training.json`, replaces peak files in `examples/TF_peaks_and_memmaped_fasta_DNASE_training.json` with processed regions and labels files. We are now ready to get started with model development and interpretation.
### Model training

### Model testing

### Interpreting data with trained models

### Processing raw peak files into intervals and labels
