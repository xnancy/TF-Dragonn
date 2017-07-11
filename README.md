# tf-dragonn
A package with command-line interface to for genome-wide, multi-modal DragoNNs.

# Usage
The `tfdragonn` command-line interface with supports interval labeling, model training and model testing. To get an overview of the interface run:
```
usage: tfdragonn <command> <args>

    The tfdragonn commands are:
    train           Train a model
    test            Test a model
    predict         Run prediction on a list of regions
    labelregions    Label a list of regions for training


TF-DragoNN command line tools

positional arguments:
  command     Subcommand to run; possible commands: test, predict, train,
              labelregions

optional arguments:
  -h, --help  show this help message and exit
```
Note: `tfdragonn predict` is still under development.

## Model Training
```
usage: tfdragonn train [-h] --visiblegpus VISIBLEGPUS [--maxexs MAXEXS]
                       [--is-tfbinding-project]
		       [--holdout-chroms HOLDOUT_CHROMS]
		       [--valid-chroms VALID_CHROMS]
		       [--learning-rate LEARNING_RATE]
		       [--batch-size BATCH_SIZE] [--epoch-size EPOCH_SIZE]
		       [--early-stopping-metric EARLY_STOPPING_METRIC]
		       [--early-stopping-patience EARLY_STOPPING_PATIENCE]
		       datasetspec intervalspec modelspec logdir

positional arguments:
  datasetspec           Dataset parameters json file path
  intervalspec          Interval parameters json file path
  modelspec             Model parameters json file path
  logdir                Log directory, also used as globally unique run
	                identifier

optional arguments:
  -h, --help            show this help message and exit
  --visiblegpus VISIBLEGPUS
                        Visible GPUs string
  --maxexs MAXEXS       max number of examples
  --is-tfbinding-project
		        Use tf-binding project specific settings
  --holdout-chroms HOLDOUT_CHROMS
			Set of chroms to holdout entirely from
			training/validation as a json string, default:
			"['chr1', 'chr8', 'chr21']"
  --valid-chroms VALID_CHROMS
			Set of chroms to holdout from training and use for
			validation as a json string, default: "['chr9']"
  --learning-rate LEARNING_RATE
			Learning rate (float), default: 0.0003
  --batch-size BATCH_SIZE
			Batch size (int), default: 256
  --epoch-size EPOCH_SIZE
			Epoch size (int), default: 2500000
  --early-stopping-metric EARLY_STOPPING_METRIC
			Early stopping metric key, default: auPRC
  --early-stopping-patience EARLY_STOPPING_PATIENCE
			Early stopping patience (int), default: 4
```

## The datasetspec file
The `datasetspec` is a json with mapping from dataset ids to data sources for each dataset. Different datasets may be different celltypes or species, and the data sources can be either genomedatalayer data directories for genome/bigwigs or bedgraphs with annotation data (such as gene expression or GENCODE annotations). Below is a the format for minimal `datasetspec` with a single dataset with a genome data source only.
```
{
    "dataset_name_maybe_mESC": {
        "genome_data_dir": "<new_output_directory_to_be_created_probably_in_/srv/scratch>"
    }
}
```
A more comprehensive example, with genome and DNase data sources for multiple celltypes, can be found in `examples/processed_sequence_dnase.json`.

## The intervalspec file
The `intervalspec` is a json with a mapping from dataset ids to intervals files. Each interval file is a tab-delimited file where the first 3 columns are `chr start end` and remaining columns are labels. An additional required `task_names` field maps to a list of label names for the labels in the intervals files. An example `intervalspec` can be found in `examples/ATF7.json`.

## Generating an intervalspec file
The `tfdragonn labelregions` command is a utility for generating an `intervalspec` from raw peaks files typically generated from data processing pipelines:
```
usage: tfdragonn labelregions [-h] [--n-jobs N_JOBS] [--bin-size BIN_SIZE]
                              [--flank-size FLANK_SIZE] [--stride STRIDE]
			      [--genome GENOME] [--logdir LOGDIR]
			      raw_intervals_config_file prefix

Generate fixed length regions and their labels for each dataset.

positional arguments:
  raw_intervals_config_file
                        Includes task names and a map from dataset id -> raw interval file
  prefix                prefix of output files

optional arguments:
  -h, --help            show this help message and exit
  --n-jobs N_JOBS       num of processes.
                        Default: 1.
  --bin-size BIN_SIZE   size of bins for labeling.
			Default: 200.
  --flank-size FLANK_SIZE
		        size of flanks around labeled bins.
			Default: 400.
  --stride STRIDE       spacing between consecutive bins.
			Default: 50.
  --genome GENOME       Genome name.
			Default: hg19.
			Options: hg18, hg38, mm9, mm10, dm3, dm6.
  --logdir LOGDIR       Logging directory
```
The `raw_intervals_config_file` is a mapping from dataset ids to universal region files, foreground region files, and (optionally) ambiguous region files:
```
{
    "task_names": ["MYC"],
    "A549": {
        "region_bed": "/mnt/lab_data/kundaje/jisraeli/projects/TF_Challenge/data/DNASE/peaks/conservative/DNASE.A549.conservative.narrowPeak.gz",
        "ambiguous_feature_beds": {"MYC": "/mnt/lab_data/kundaje/jisraeli/projects/TF_Challenge/data/ChIPseq/peaks/relaxed/ChIPseq.A549.MYC.relaxed.narrowPeak.gz"
	},
	"feature_beds": {"MYC": "/mnt/lab_data/kundaje/jisraeli/projects/TF_Challenge/data/ChIPseq/peaks/conservative/ChIPseq.A549.MYC.conservative.train.narrowPeak.gz"
	}
    },
}
```
where `region_bed` is the universal regions file (for example DNase peaks or full genome), `feature_beds` is a mapping from task names to foreground regions for each task, and `ambiguous_feature_beds` is a mapping from task names to ambiguous regions for each task.

## The modelspec file
The `modelspec` file specifies the model architecture for training:
```
{
    "model_class": "SequenceClassifier",
     "num_filters": [30, 30, 30]
}
```
A required `model_class` field specifies a model class from `models.py` and the remaining fields specify argument for the constructor of that class. You can implement your own model classes in `models.py`.