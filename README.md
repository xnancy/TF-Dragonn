# tf-dragonn
Reimplementing tf binding with simpler, faster, and more stable code. Basic workflow:

## Usage
I'm in the process of developing a `tfdragonn` command-line interface. The basic idea is to provide a minimal set of commands for extensive model development and interpretation, with bed intervals as the main interface, and minimal usage of data/model config files. Here are examples of what can be done so far:
```
tfdragonn train --data-config-file examples/hydrogel_data_config.json --prefix example_cmd_line_run
```
This will train a sequence-only model using the genome fasta and the union of feature bed regions in hydrogel_data_config.json and store output files based on the prefix.

## Roadmap
0. tf/celltype name -> raw peaks files and signal files (using our database, for internal use only)
    * Preliminary support in dev branch
1. raw peaks files -> regions & labels
    * Available in intervals.py
    * regions & scores (for regression) in dev branch
2. regions + signal files -> memmapped data w streaming (large scale data) or data arrays in memory (small scale data)
    * Minimal command-line interface for sequence-only
    * TODO:
	* add bigwig extraction using genomedatalayer
	* add utilties for multi-sample data organization
        * added memmapped/streaming data option using genomedatalayer
	* port to tensorflow using genomeflow
3. data + labels -> trained model
    * SequenceClassifier available in models.py using keras
    * TODO:
        * add sequence+dnase classification model
        * port to tensorflow
4. postprocessing (bigwigs with scoress, motif clustering)
    * Preliminary utilities available in postprocessing_utils.py
    * TODO
        * comparison to known motifs
	* visualization of known motif and clustered grammar hits in browser
	* ranking/prioratization of clustered grammars
        * support tensorflow models by portin deeplift to tf or porting tensorflow models to keras
