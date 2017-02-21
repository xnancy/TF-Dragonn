from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty, ABCMeta
from builtins import zip
import json
import numpy as np
import sys

from keras.layers import (
    Activation, AveragePooling1D, Convolution1D, Dense,
    Dropout, Flatten, Input, MaxPooling1D, Merge, Permute,
    Reshape
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


class SequenceAndDnaseClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir", "data/dnase_data_dir"]

    def __init__(self, interval_size, num_tasks,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 pool_width=25,
                 fc_layer_widths=(),
                 seq_conv_dropout=0,
                 dnase_conv_dropout=0,
                 combined_conv_dropout=0,
                 fc_layer_dropout=0):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        # convolve sequence
        seq_inputs = Input(shape=(4, interval_size), name="data/genome_data_dir")
        seq_preds = seq_inputs
        seq_preds = Permute((2, 1))(seq_preds) # conv1d expects (interval_size, 4)
        for nb_filter, nb_col in zip(num_seq_filters, seq_conv_width):
            seq_preds = Convolution1D(nb_filter, nb_col, 'he_normal')(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if seq_conv_dropout > 0:
                seq_preds = Dropout(dropout)(seq_preds)

        # convolve dnase
        dnase_inputs = Input(shape=(interval_size,), name="data/dnase_data_dir")
        dnase_preds = dnase_inputs
        dnase_preds = Reshape((1000, 1))(dnase_preds) # conv1d expects (interval_size, 1)
        for nb_filter, nb_col in zip(num_dnase_filters, dnase_conv_width):
            dnase_preds = Convolution1D(nb_filter, nb_col, 'he_normal')(dnase_preds)
            dnase_preds = Activation('relu')(dnase_preds)
            if dnase_conv_dropout > 0:
                dnase_preds = Dropout(dropout)(dnase_preds)

        # stack and convolve
        logits  = Merge(mode='concat', concat_axis=-1)([seq_preds, dnase_preds])
        for nb_filter, nb_col in zip(num_combined_filters, combined_conv_width):
            logits = Convolution1D(nb_filter, nb_col, 'he_normal')(logits)
            logits = Activation('relu')(logits)
            if combined_conv_dropout > 0:
                logits = Dropout(dropout)(logits)

        # pool and fully connect
        logits = AveragePooling1D((pool_width))(logits)
        logits = Flatten()(logits)
        for fc_layer_width in fc_layer_widths:
            logits = Dense(fc_layer_width)(logits)
            logits = Activation('relu')(logits)
            if fc_layer_dropout > 0:
                logits = Dropout(dropout)(logits)
        logits = Dense(num_tasks)(logits)
        logits = Activation('sigmoid')(logits)
        self.model = Model(input=[seq_inputs, dnase_inputs], output=logits)
