from collections import namedtuple
import os.path

import model_runner

DEFAULT_HOLDOUT_CHROMS = {'chr1', 'chr8', 'chr21'}
DEFAULT_VALID_CHROMS = {'chr9'}

DEFAULT_EARLYSTOPPING_KEY = 'auPRC'
DEFAULT_EARLYSTOPPING_PATIENCE = 4

IN_MEMORY = False
BATCH_SIZE = 256
EPOCH_SIZE = 2500000
LEARNING_RATE = 0.0003


"""
Parameters for a model run. Format:
{'param_name': (type, is_required (bool), default_value, help_str)}
"""
ModelRunParamsSpec = [
    ('datasetspec', (os.path.abspath, True, None, 'Dataset parameters json file path')),
    ('intervalspec', (os.path.abspath, True, None, 'Interval parameters json file path')),
    ('modelspec', (os.path.abspath, True, None, 'Model parameters json file path')),
    ('logdir', (os.path.abspath, True, None, 'Log directory')),
    ('maxexs', (int, False, None, 'Max number of examples')),
    ('visiblegpus', (str, True, None, 'Visible GPUs string')),
    ('is_tfbinding_project', (bool, False, False, 'Use tf-binding project specific settings'))
]
keys = [p[0] for p in ModelRunParamsSpec]
assert(len(keys) == len(set(keys)))

TrainModelRunParamsSpec = [
    ('holdout_chroms', (set, False, model_runner.DEFAULT_HOLDOUT_CHROMS,
                        'Test chroms to holdout from training/validation')),
    ('valid_chroms', (set, False, model_runner.DEFAULT_VALID_CHROMS,
                      'Validation to holdout from training and use for validation')),
    ('learning_rate', (float, False, model_runner.DEFAULT_LEARNING_RATE, 'Learning rate')),
    ('batch_size', (int, False, model_runner.DEFAULT_BATCH_SIZE, 'Batch size')),
    ('epoch_size', (int, False, model_runner.DEFAULT_EPOCH_SIZE, 'Epoch size')),
    ('early_stopping_metric', (str, False, model_runner.DEFAULT_EARLYSTOPPING_KEY, 'Early stopping metric key')),
    ('early_stopping_patience', (int, False, model_runner.DEFAULT_EARLYSTOPPING_PATIENCE, 'Early stopping patience')),
]
TrainModelRunParamsSpec = ModelRunParamsSpec + TrainModelRunParamsSpec
keys = [p[0] for p in TrainModelRunParamsSpec]
assert(len(keys) == len(set(keys)))


# Container for storing model run params
ModelRunParams = namedtuple('ModelRunParams', [k for k in ModelRunParamsSpec])
TrainingParams = namedtuple('TrainingParams', [k for k in TrainModelRunParamsSpec])


def get_params_from_lookup(params_class, params_list, params_lookup):
    params = []
    for (param_name, (param_type, param_required, param_default, param_help)) in params_list:
        if param_required and param_name not in params_lookup:
            raise ValueError('{} ({}) is required for ModelRunnerParams'.format(
                param_name, param_help))
        if param_name in params_lookup:
            params.append(param_type(params_lookup[param_name]))
        else:
            params.append(param_default)
    return params_class(*params)


def get_model_run_params(**kwargs):
    return get_params_from_lookup(ModelRunParams, ModelRunParamsSpec, kwargs)


def get_training_params(**kwargs):
    return get_params_from_lookup(TrainingParams, TrainModelRunParamsSpec, kwargs)
