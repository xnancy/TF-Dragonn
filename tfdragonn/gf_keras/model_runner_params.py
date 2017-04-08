from collections import namedtuple
import os.path

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
ModelRunParams = {
    'datasetspec': (os.path.abspath, True, None, 'Dataset parameters json file path'),
    'intervalspec': (os.path.abspath, True, None, 'Interval parameters json file path'),
    'modelspec': (os.path.abspath, True, None, 'Model parameters json file path'),
    'logdir': (os.path.abspath, True, None, 'Log directory'),
    'numexs': (int, False, None, 'Max number of examples'),
    'holdout_chroms': (set, False, DEFAULT_HOLDOUT_CHROMS,
                       'Test chroms to holdout from training/validation'),
    'valid_chroms': (set, False, DEFAULT_VALID_CHROMS,
                     'Validation to holdout from training and use for validation')
    'early_stopping_metric': (str, False, DEFAULT_EARLYSTOPPING_KEY, 'Early stopping metric key'),

}

# Container for storing model run params
ModelRunParams = namedtuple(
    'ModelRunParams',
    ['datasetspec', 'intervalspec', 'modelspec', 'logdir', 'numexs', 'holdout_chroms', 'valid_chroms', 'batch_size'],
)


def get_model_run_params(datasetspec, intervalspec, modelspec, logdir, numexs=None):
    datasetspec = os.path.abspath(datasetspec)
    intervalspec = os.path.abspath(intervalspec)
    modelspec = os.path.abspath(modelspec)
    logdir = os.path.abspath(logdir)
    return ModelRunParams(datasetspec, intervalspec, modelspec, logdir, numexs)
