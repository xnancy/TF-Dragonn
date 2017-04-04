from collections import namedtuple
import os.path

# Container for storing model run params
ModelRunParams = namedtuple(
    'ModelRunParams',
    ['datasetspec', 'intervalspec', 'modelspec', 'logdir', 'numexs'],
)


def get_model_run_params(datasetspec, intervalspec, modelspec, logdir, numexs=None):
    datasetspec = os.path.abspath(datasetspec)
    intervalspec = os.path.abspath(intervalspec)
    modelspec = os.path.abspath(modelspec)
    logdir = os.path.abspath(logdir)
    return ModelRunParams(datasetspec, intervalspec, modelspec, logdir, numexs)
