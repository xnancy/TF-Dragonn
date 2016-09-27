# tf-dragonn
Reimplementing tf binding with simpler, faster, and more stable code.

## Usage
I'm in the process of developing a `tfdragonn` command-line interface. The basic idea is to provide a minimal set of commands for extensive model development and interpretation, with bed intervals as the main interface, and minimal usage of data/model config files. Here are examples of what can be done so far:
```
tfdragonn train --data-config-file examples/hydrogel_data_config.json --prefix example_cmd_line_run
```
This will train a sequence-only model using the genome fasta and the union of feature bed regions in hydrogel_data_config.json and store output files based on the prefix.