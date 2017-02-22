from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import sys

from keras import backend as K
from keras.layers import (
    Activation, AveragePooling1D, BatchNormalization,
    Convolution1D, Dense, Dropout, Flatten, Input,
    MaxPooling1D, Merge, Permute, Reshape
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


def model_from_config_and_queue(model_config_file_path, queue):
    """
    Uses queue output shapes and json file
    with architecture params in to initialize a model
    """
    thismodule = sys.modules[__name__]
    with open(model_config_file_path, 'r') as fp:
        config = json.load(fp)
    model_class_name = config['model_class']

    model_class = getattr(thismodule, model_class_name)
    del config['model_class']
    return model_class(queue.output_shapes, **config)


def reshape_dnase_input(x):
    """
    Reshapes (interval_size) shaped dnase data
    from example queues into (interval_size, 1)
    """
    interval_size = K.int_shape(x)[-1]
    return Reshape((interval_size, 1))(x)


_input_reshape_func = {
    # conv1d expects (interval_size, 4)
    "data/genome_data_dir": Permute((2, 1)),
    "data/dnase_data_dir": reshape_dnase_input
}


model_inputs = {
    "SequenceClassifier": [
        "data/genome_data_dir"],
    "SequenceAndDnaseClassifier": [
        "data/genome_data_dir",
        "data/dnase_data_dir"],
    "SequenceDnaseTssDhsCountAndTssExpressionClassifier": [
        "data/genome_data_dir",
        "data/dnase_data_dir",
        "data/dhs_counts",
        "data/tss_counts",
        "data/tss_mean_tpm",
        "data/tss_max_tpm"]
}


def model_inputs_from_config(model_config_file_path):
    thismodule = sys.modules[__name__]
    with open(model_config_file_path, 'r') as fp:
        config = json.load(fp)
    return model_inputs[config['model_class']]


class Classifier(object):
    """
    Classifier interface.

    Args:
        shapes (dict): a dict of input/output shapes.
            Example: `{data/genome_data_dir: (4, 1000)}`

    Attributes:
        get_inputs (list): a list of input names.
            Derived from model_inputs unless implemented.
    """
    @property
    def get_inputs(self):
        return model_inputs[self.__class__.__name__]

    def __init__(self, shapes, **hyperparameters):
        pass

    def save(self, prefix):
        arch_fname = prefix + '.arch.json'
        weights_fname = prefix + '.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)

    def get_keras_inputs(self, shapes):
        """Returns dictionary of named keras inputs"""
        return collections.OrderedDict(
            [(name, Input(shape=shapes[name], name=name))
             for name in self.get_inputs])

    @staticmethod
    def reshape_keras_inputs(keras_inputs):
        """reshapes keras inputs based on example queues"""
        inputs = collections.OrderedDict()
        for k, v in keras_inputs.items():
            if k in _input_reshape_func:  # reshape
                inputs[k] = _input_reshape_func[k](v)
            else:  # keep as is
                inputs[k] = v
        return inputs


class SequenceClassifier(Classifier):

    def __init__(self, shapes,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 pool_width=35, dropout=0, batch_norm=False):
        assert len(num_filters) == len(conv_width)

        # configure inputs
        keras_inputs = self.get_keras_inputs(shapes)
        inputs = self.reshape_keras_inputs(keras_inputs)

        # convolve sequence
        seq_preds = inputs["data/genome_data_dir"]
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            seq_preds = Convolution1D(
                nb_filter, nb_col, 'he_normal')(seq_preds)
            if batch_norm:
                seq_preds = BatchNormalization()(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if dropout > 0:
                seq_preds = Dropout(dropout)(seq_preds)

        # pool and fully connect
        seq_preds = AveragePooling1D((pool_width))(seq_preds)
        seq_preds = Flatten()(seq_preds)
        seq_preds = Dense(output_dim=shapes['labels'][-1])(seq_preds)
        seq_preds = Activation('sigmoid')(seq_preds)
        self.model = Model(input=keras_inputs.values(), output=seq_preds)


class SequenceAndDnaseClassifier(Classifier):

    def __init__(self, shapes,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 pool_width=25,
                 fc_layer_widths=(100,),
                 seq_conv_dropout=0.1,
                 dnase_conv_dropout=0.1,
                 combined_conv_dropout=0.1,
                 fc_layer_dropout=0.1,
                 batch_norm=False):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        # configure inputs
        keras_inputs = self.get_keras_inputs(shapes)
        inputs = self.reshape_keras_inputs(keras_inputs)

        # convolve sequence
        seq_preds = inputs["data/genome_data_dir"]
        for nb_filter, nb_col in zip(num_seq_filters, seq_conv_width):
            seq_preds = Convolution1D(
                nb_filter, nb_col, 'he_normal')(seq_preds)
            if batch_norm:
                seq_preds = BatchNormalization()(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if seq_conv_dropout > 0:
                seq_preds = Dropout(seq_conv_dropout)(seq_preds)

        # convolve dnase
        dnase_preds = inputs["data/dnase_data_dir"]
        for nb_filter, nb_col in zip(num_dnase_filters, dnase_conv_width):
            dnase_preds = Convolution1D(
                nb_filter, nb_col, 'he_normal')(dnase_preds)
            if batch_norm:
                dnase_preds = BatchNormalization()(dnase_preds)
            dnase_preds = Activation('relu')(dnase_preds)
            if dnase_conv_dropout > 0:
                dnase_preds = Dropout(dnase_conv_dropout)(dnase_preds)

        # stack and convolve
        logits = Merge(mode='concat', concat_axis=-1)([seq_preds, dnase_preds])
        for nb_filter, nb_col in zip(num_combined_filters, combined_conv_width):
            logits = Convolution1D(nb_filter, nb_col, 'he_normal')(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if combined_conv_dropout > 0:
                logits = Dropout(combined_conv_dropout)(logits)

        # pool and fully connect
        logits = AveragePooling1D((pool_width))(logits)
        logits = Flatten()(logits)
        for fc_layer_width in fc_layer_widths:
            logits = Dense(fc_layer_width)(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if fc_layer_dropout > 0:
                logits = Dropout(fc_layer_dropout)(logits)
        logits = Dense(shapes['labels'][-1])(logits)
        logits = Activation('sigmoid')(logits)
        self.model = Model(input=keras_inputs.values(), output=logits)


class SequenceDnaseTssDhsCountAndTssExpressionClassifier(Classifier):

    def __init__(self, shapes,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 pool_width=25,
                 seq_dnase_fc_layer_widths=(100,),
                 final_fc_layer_widths=(100,),
                 seq_conv_dropout=0.1,
                 dnase_conv_dropout=0.1,
                 combined_conv_dropout=0.1,
                 seq_dnase_fc_layer_dropout=0.1,
                 final_fc_dropout=0.1,
                 batch_norm=False):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        # configure inputs
        keras_inputs = self.get_keras_inputs(shapes)
        inputs = self.reshape_keras_inputs(keras_inputs)

        # convolve sequence
        seq_preds = inputs["data/genome_data_dir"]
        for nb_filter, nb_col in zip(num_seq_filters, seq_conv_width):
            seq_preds = Convolution1D(
                nb_filter, nb_col, 'he_normal')(seq_preds)
            if batch_norm:
                seq_preds = BatchNormalization()(seq_preds)
            seq_preds = Activation('relu')(seq_preds)
            if seq_conv_dropout > 0:
                seq_preds = Dropout(seq_conv_dropout)(seq_preds)

        # convolve dnase
        dnase_preds = inputs["data/dnase_data_dir"]
        for nb_filter, nb_col in zip(num_dnase_filters, dnase_conv_width):
            dnase_preds = Convolution1D(
                nb_filter, nb_col, 'he_normal')(dnase_preds)
            if batch_norm:
                dnase_preds = BatchNormalization()(dnase_preds)
            dnase_preds = Activation('relu')(dnase_preds)
            if dnase_conv_dropout > 0:
                dnase_preds = Dropout(dnase_conv_dropout)(dnase_preds)

        # stack sequence + dnase and convolve
        logits = Merge(mode='concat', concat_axis=-1)([seq_preds, dnase_preds])
        for nb_filter, nb_col in zip(num_combined_filters, combined_conv_width):
            logits = Convolution1D(nb_filter, nb_col, 'he_normal')(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if combined_conv_dropout > 0:
                logits = Dropout(combined_conv_dropout)(logits)

        # pool and fully connect seq + dnase
        logits = AveragePooling1D((pool_width))(logits)
        logits = Flatten()(logits)
        for fc_layer_width in seq_dnase_fc_layer_widths:
            logits = Dense(fc_layer_width)(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if seq_dnase_fc_layer_dropout > 0:
                logits = Dropout(seq_dnase_fc_layer_dropout)(logits)

        # merge in tss+dhs counts, tss tpms and fully connected
        logits = Merge(mode='concat', concat_axis=-1)([
            logits, inputs['data/dhs_counts'], inputs['data/tss_counts'],
            inputs['data/tss_mean_tpm'], inputs['data/tss_max_tpm']])
        for fc_layer_width in final_fc_layer_widths:
            logits = Dense(fc_layer_width)(logits)
            if batch_norm:
                logits = BatchNormalization()(logits)
            logits = Activation('relu')(logits)
            if final_fc_dropout > 0:
                logits = Dropout(final_fc_dropout)(logits)

        logits = Dense(shapes['labels'][-1])(logits)
        logits = Activation('sigmoid')(logits)
        self.model = Model(input=keras_inputs.values(), output=logits)
