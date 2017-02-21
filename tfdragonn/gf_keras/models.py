from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty, ABCMeta
from builtins import zip
import json
import numpy as np
import sys

from keras.layers import (
    Convolution1D, Input, MaxPooling1D, AveragePooling1D,
    Activation, Dense, Dropout, Flatten, Permute
)
from keras.models import Model

def model_from_config(model_config_file_path):
    """Load a model from a json config file."""
    thismodule = sys.modules[__name__]
    with open(model_config_file_path, 'r') as fp:
        config = json.load(fp)
    model_class_name = config['model_class']

    model_class = getattr(thismodule, model_class_name)
    del config['model_class']
    return model_class(**config)


class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def get_inputs(self):
        pass

    @abstractmethod
    def __init__(self, **hyperparameters):
        pass

    def save(self, prefix):
        arch_fname = prefix + '.arch.json'
        weights_fname = prefix + '.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)


class SequenceClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir"]

    def __init__(self, interval_size, num_tasks,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 pool_width=35, dropout=0):
        assert len(num_filters) == len(conv_width)

        self.num_tasks = num_tasks
        seq_inputs = Input(shape=(4, interval_size), name="data/genome_data_dir")
        seq_preds = seq_inputs
        seq_preds = Permute((2, 1))(seq_preds) # conv1d expects (interval_size, 4)
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            seq_preds = Convolution1D(nb_filter, nb_col, 'he_normal')(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if dropout > 0:
                seq_preds = Dropout(dropout)(seq_preds)
        seq_preds = AveragePooling1D((pool_width))(seq_preds)
        seq_preds = Flatten()(seq_preds)
        seq_preds = Dense(output_dim=num_tasks)(seq_preds)
        seq_preds = Activation('sigmoid')(seq_preds)
        self.model = Model(input=seq_inputs, output=seq_preds)
