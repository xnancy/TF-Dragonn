from __future__ import absolute_import, division, print_function

import json
import sys

from abc import abstractmethod, abstractproperty, ABCMeta
import tensorflow as tf
import tensorflow.contrib.slim as slim

import initializers


def model_from_config(model_config_file_path):
    """Load a model from a json config file."""
    # TODO(jisraeli): this should support loading model parameters

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

    @abstractmethod
    def get_logits(inputs):
        pass


def expand_4D(input_tensor):
    shape = [x.value for x in input_tensor.get_shape()]
    if len(shape) == 2:  # 1-D input
        new_shape = [shape[0], 1, shape[1], 1]
    elif len(shape) == 3:  # 2-D input
        new_shape = shape + [1]
    else:
        raise IOError('unrecognized shape: {}'.format(shape))
    return tf.reshape(input_tensor, new_shape)


class SequenceClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir"]

    def __init__(self,
                 num_filters=(25, 25, 25), conv_width=(25, 25, 25),
                 pool_width=25, fc_layer_widths=(500,),
                 task_specific_fc_layer_widths=(80,),
                 conv_dropout=0,
                 fc_layer_dropout=0.2,
                 batch_norm=True):
        assert len(num_filters) == len(conv_width)

        self.num_filters = num_filters
        self.conv_width = conv_width
        self.pool_width = pool_width
        self.fc_layer_widths = fc_layer_widths
        self.task_specific_fc_layer_widths = task_specific_fc_layer_widths
        self.conv_dropout = conv_dropout
        self.fc_layer_dropout = fc_layer_dropout
        self.batch_norm = batch_norm

    def get_logits(self, inputs, num_tasks):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected], reuse=False, activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm if self.batch_norm else None,
                weights_initializer=initializers.he_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0)):

            seq_preds = inputs["data/genome_data_dir"]
            seq_preds = expand_4D(seq_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_filters, self.conv_width)):
                filter_height = 4 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                seq_preds = slim.conv2d(seq_preds, num_filter, filter_dims, padding='VALID',
                                        scope='sequence_conv{:d}'.format(i + 1))
                if self.conv_dropout > 0:
                    seq_preds = slim.dropout(seq_preds, self.conv_dropout)

            print('shape after conv layers, before pooling: {}'.format(seq_preds.get_shape()))
            seq_preds = slim.avg_pool2d(seq_preds, [1, self.pool_width], stride=[1, self.pool_width],
                                     padding='VALID', scope='avg_pool')
            seq_preds = slim.flatten(seq_preds, scope='flatten')
            for i, fc_layer_width in enumerate(self.fc_layer_widths):
                seq_preds = slim.fully_connected(seq_preds, fc_layer_width, scope='fc{}'.format(i + 1))
                if self.fc_layer_dropout > 0:
                    seq_preds = slim.dropout(seq_preds, self.fc_layer_dropout)

            if len(self.task_specific_fc_layer_widths) > 0:
                task_specific_seq_preds = []
                for task_id in xrange(num_tasks):
                    task_specific_seq_preds.append(
                        slim.stack(seq_preds, slim.fully_connected, self.task_specific_fc_layer_widths,
                                   scope='fc_task{}'.format(task_id)))
                    task_specific_seq_preds[-1] = slim.fully_connected(
                        task_specific_seq_preds[-1], 1, activation_fn=None, normalizer_fn=None,
                        scope='logit{}'.format(task_id))
                logits = tf.concat(1, task_specific_seq_preds)
            else:
                logits = slim.fully_connected(
                    seq_preds, num_tasks, activation_fn=None, normalizer_fn=None, scope='output-fc')

            return logits


class SequenceAndDnaseClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir", "data/dnase_data_dir"]

    def __init__(self,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 pool_width=25,
                 fc_layer_widths=(100,),
                 task_specific_fc_layer_widths=(),
                 seq_conv_dropout=0,
                 dnase_conv_dropout=0,
                 combined_conv_dropout=0,
                 fc_layer_dropout=0.2,
                 batch_norm=True):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        self.num_seq_filters = num_seq_filters
        self.seq_conv_width = seq_conv_width
        self.num_dnase_filters = num_dnase_filters
        self.dnase_conv_width = dnase_conv_width
        self.num_combined_filters = num_combined_filters
        self.combined_conv_width = combined_conv_width
        self.fc_layer_widths = fc_layer_widths
        self.task_specific_fc_layer_widths = task_specific_fc_layer_widths
        self.pool_width = pool_width
        self.seq_conv_dropout = seq_conv_dropout
        self.dnase_conv_dropout = dnase_conv_dropout
        self.combined_conv_dropout = combined_conv_dropout
        self.fc_layer_dropout = fc_layer_dropout
        self.batch_norm = batch_norm

    def get_logits(self, inputs, num_tasks):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected], reuse=False, activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm if self.batch_norm else None,
                weights_initializer=initializers.he_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0)):

            seq_preds = inputs["data/genome_data_dir"]
            seq_preds = expand_4D(seq_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_seq_filters, self.seq_conv_width)):
                filter_height = 4 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                seq_preds = slim.conv2d(seq_preds, num_filter, filter_dims, padding='VALID',
                                        scope='sequence_conv{:d}'.format(i + 1))
                if self.seq_conv_dropout > 0:
                    seq_preds = slim.dropout(seq_preds, self.seq_conv_dropout)

            dnase_preds = inputs["data/dnase_data_dir"]
            dnase_preds = expand_4D(dnase_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_dnase_filters, self.dnase_conv_width)):
                fitler_dims = [1, filter_width]
                dnase_preds = slim.conv2d(dnase_preds, num_filter, fitler_dims, padding='VALID',
                                          scope='dnase_conv{:d}'.format(i + 1))
                if self.dnase_conv_dropout > 0:
                    dnase_preds = slim.dropout(dnase_preds, self.combined_conv_dropout)

            # check if concatenation axis is correct
            print('seq_preds shape {}'.format(seq_preds.get_shape()))
            print('dnase_preds shape {}'.format(dnase_preds.get_shape()))
            logits = tf.concat(1, [seq_preds, dnase_preds])
            print('concat shape {}'.format(logits.get_shape()))
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_combined_filters, self.combined_conv_width)):
                filter_height = 2 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                logits = slim.conv2d(logits, num_filter, filter_dims, padding='VALID',
                                     scope='combined_conv{:d}'.format(i + 1))
                if self.combined_conv_dropout > 0:
                    logits = slim.dropout(logits, self.combined_conv_dropout)

            print('after combined conv layers shape {}'.format(logits.get_shape()))
            logits = slim.avg_pool2d(logits, [1, self.pool_width], stride=[1, self.pool_width],
                                     padding='VALID', scope='avg_pool')
            logits = slim.flatten(logits, scope='flatten')
            for i, fc_layer_width in enumerate(self.fc_layer_widths):
                logits = slim.fully_connected(logits, fc_layer_width, scope='fc{}'.format(i + 1))
                if self.fc_layer_dropout > 0:
                    logits = slim.dropout(logits, self.fc_layer_dropout)

            if len(self.task_specific_fc_layer_widths) > 0:
                task_specific_logits = []
                for task_id in xrange(num_tasks):
                    task_specific_logits.append(
                        slim.stack(logits, slim.fully_connected, self.task_specific_fc_layer_widths,
                                   scope='fc_task{}'.format(task_id)))
                    task_specific_logits[-1] = slim.fully_connected(
                        task_specific_logits[-1], 1, activation_fn=None, normalizer_fn=None,
                        scope='logit{}'.format(task_id))
                logits = tf.concat(1, task_specific_logits)
            else:
                logits = slim.fully_connected(
                    logits, num_tasks, activation_fn=None, normalizer_fn=None, scope='output-fc')

            return logits


class SequenceDnaseAndDnasePeaksCountsClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir",
                "data/dnase_data_dir",
                "data/dnase_peaks_counts_data_dir"]

    def __init__(self,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 peaks_counts_fc_layer_widths=(20,),
                 final_fc_layer_widths=(), pool_width=25, batch_norm=False):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        self.num_seq_filters = num_seq_filters
        self.seq_conv_width = seq_conv_width
        self.num_dnase_filters = num_dnase_filters
        self.dnase_conv_width = dnase_conv_width
        self.num_combined_filters = num_combined_filters
        self.combined_conv_width = combined_conv_width
        self.peaks_counts_fc_layer_widths = peaks_counts_fc_layer_widths
        self.final_fc_layer_widths = final_fc_layer_widths
        self.pool_width = pool_width
        self.batch_norm = batch_norm

    def get_logits(self, inputs, num_tasks):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected], reuse=False, activation_fn=tf.nn.relu,
                weights_initializer=initializers.he_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0)):

            seq_preds = inputs["data/genome_data_dir"]
            seq_preds = expand_4D(seq_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_seq_filters, self.seq_conv_width)):
                filter_height = 4 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                seq_preds = slim.conv2d(seq_preds, num_filter, filter_dims, padding='VALID',
                                        scope='sequence_conv{:d}'.format(i + 1))

            dnase_preds = inputs["data/dnase_data_dir"]
            dnase_preds = expand_4D(dnase_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_dnase_filters, self.dnase_conv_width)):
                fitler_dims = [1, filter_width]
                dnase_preds = slim.conv2d(dnase_preds, num_filter, fitler_dims, padding='VALID',
                                          scope='dnase_conv{:d}'.format(i + 1))

            # check if concatenation axis is correct
            print('seq_preds shape {}'.format(seq_preds.get_shape()))
            print('dnase_preds shape {}'.format(dnase_preds.get_shape()))
            seq_dnase_preds = tf.concat(1, [seq_preds, dnase_preds])
            print('concat shape {}'.format(seq_dnase_preds.get_shape()))
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_combined_filters, self.combined_conv_width)):
                filter_height = 2 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                seq_dnase_preds = slim.conv2d(seq_dnase_preds, num_filter, filter_dims, padding='VALID',
                                              scope='combined_conv{:d}'.format(i + 1))
            print('after combined conv layers shape {}'.format(seq_dnase_preds.get_shape()))
            seq_dnase_preds = slim.avg_pool2d(seq_dnase_preds, [1, self.pool_width], stride=[1, self.pool_width],
                                              padding='VALID', scope='avg_pool')
            seq_dnase_preds = slim.flatten(seq_dnase_preds, scope='flatten')
            print('after flattening sequence and dnase shape {}'.format(seq_dnase_preds.get_shape()))

            # fully connect dnase peaks counts and concatenate
            peaks_counts_preds = inputs["data/dnase_peaks_counts_data_dir"]
            peaks_counts_preds = slim.flatten(peaks_counts_preds, scope='flatten_peaks_counts')
            print('peaks_counts_preds before fc layers shape {}'.format(peaks_counts_preds.get_shape()))
            if len(self.peaks_counts_fc_layer_widths) > 0:
                peaks_counts_preds = slim.stack(peaks_counts_preds, slim.fully_connected,
                                                self.peaks_counts_fc_layer_widths, scope='fc_peaks_counts')
            print('peaks_counts_preds after fc layers shape {}'.format(peaks_counts_preds.get_shape()))
            logits = tf.concat(1, [seq_dnase_preds, peaks_counts_preds])
            print('concat peaks_counts_preds and seq_dnase_preds shape {}'.format(logits.get_shape()))

            # fully connect everything
            if len(self.final_fc_layer_widths) > 0:
                logits = slim.stack(logits, slim.fully_connected, self.final_fc_layer_widths, scope='fc')
            logits = slim.fully_connected(
                logits, num_tasks, activation_fn=None, scope='output-fc')

            return logits


class SequenceDnaseDnasePeaksCountsAndGencodeClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir",
                "data/dnase_data_dir",
                "data/dnase_peaks_counts_data_dir",
                "data/gencode_tss_distances_data_dir",
                "data/gencode_annotation_distances_data_dir",
                "data/gencode_polyA_distances_data_dir",
                "data/gencode_lncRNA_distances_data_dir"]

    def __init__(self,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 peaks_counts_fc_layer_widths=(20,), pre_gencode_fc_layer_widths=(50,),
                 gencode_fc_layer_widths=(30,),
                 final_fc_layer_widths=(), pool_width=25, batch_norm=False):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        self.num_seq_filters = num_seq_filters
        self.seq_conv_width = seq_conv_width
        self.num_dnase_filters = num_dnase_filters
        self.dnase_conv_width = dnase_conv_width
        self.num_combined_filters = num_combined_filters
        self.combined_conv_width = combined_conv_width
        self.peaks_counts_fc_layer_widths = peaks_counts_fc_layer_widths
        self.pre_gencode_fc_layer_widths = pre_gencode_fc_layer_widths  # everything but gencode features
        self.gencode_fc_layer_widths = gencode_fc_layer_widths  # gencode features only
        self.final_fc_layer_widths = final_fc_layer_widths  # everything
        self.pool_width = pool_width
        self.batch_norm = batch_norm

    def get_logits(self, inputs, num_tasks):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected], reuse=False, activation_fn=tf.nn.relu,
                weights_initializer=initializers.he_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0)):

            seq_preds = inputs["data/genome_data_dir"]
            seq_preds = expand_4D(seq_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_seq_filters, self.seq_conv_width)):
                filter_height = 4 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                seq_preds = slim.conv2d(seq_preds, num_filter, filter_dims, padding='VALID',
                                        scope='sequence_conv{:d}'.format(i + 1))

            dnase_preds = inputs["data/dnase_data_dir"]
            dnase_preds = expand_4D(dnase_preds)
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_dnase_filters, self.dnase_conv_width)):
                fitler_dims = [1, filter_width]
                dnase_preds = slim.conv2d(dnase_preds, num_filter, fitler_dims, padding='VALID',
                                          scope='dnase_conv{:d}'.format(i + 1))

            # check if concatenation axis is correct
            print('seq_preds shape {}'.format(seq_preds.get_shape()))
            print('dnase_preds shape {}'.format(dnase_preds.get_shape()))
            seq_dnase_preds = tf.concat(1, [seq_preds, dnase_preds])
            print('concat shape {}'.format(seq_dnase_preds.get_shape()))
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_combined_filters, self.combined_conv_width)):
                filter_height = 2 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                seq_dnase_preds = slim.conv2d(seq_dnase_preds, num_filter, filter_dims, padding='VALID',
                                              scope='combined_conv{:d}'.format(i + 1))
            print('after combined conv layers shape {}'.format(seq_dnase_preds.get_shape()))
            seq_dnase_preds = slim.avg_pool2d(seq_dnase_preds, [1, self.pool_width], stride=[1, self.pool_width],
                                              padding='VALID', scope='avg_pool')
            seq_dnase_preds = slim.flatten(seq_dnase_preds, scope='flatten')
            print('after flattening sequence and dnase shape {}'.format(seq_dnase_preds.get_shape()))

            # fully connect dnase peaks counts and concatenate
            peaks_counts_preds = inputs["data/dnase_peaks_counts_data_dir"]
            peaks_counts_preds = slim.flatten(peaks_counts_preds, scope='flatten_peaks_counts')
            print('peaks_counts_preds before fc layers shape {}'.format(peaks_counts_preds.get_shape()))
            if len(self.peaks_counts_fc_layer_widths) > 0:
                peaks_counts_preds = slim.stack(peaks_counts_preds, slim.fully_connected,
                                                self.peaks_counts_fc_layer_widths, scope='fc_peaks_counts')
            print('peaks_counts_preds after fc layers shape {}'.format(peaks_counts_preds.get_shape()))
            logits = tf.concat(1, [seq_dnase_preds, peaks_counts_preds])
            print('concat peaks_counts_preds and seq_dnase_preds shape {}'.format(logits.get_shape()))
            if len(self.pre_gencode_fc_layer_widths) > 0:
                logits = slim.stack(logits, slim.fully_connected, self.pre_gencode_fc_layer_widths, scope='fc_pre_gencode')

            # fully connect gencode features
            gencode_preds = tf.concat(1, [inputs["data/gencode_tss_distances_data_dir"],
                                          inputs["data/gencode_annotation_distances_data_dir"],
                                          inputs["data/gencode_polyA_distances_data_dir"],
                                          inputs["data/gencode_lncRNA_distances_data_dir"]])
            gencode_preds = tf.squeeze(gencode_preds)  # remove extranous dimension
            gencode_preds = tf.log(1 + gencode_preds)  # take log of distances
            if len(self.gencode_fc_layer_widths) > 0:
                gencode_preds = slim.stack(gencode_preds, slim.fully_connected, self.gencode_fc_layer_widths, scope='fc_gencode')

            # fully connect everything
            logits = tf.concat(1, [logits, gencode_preds])
            if len(self.final_fc_layer_widths) > 0:
                logits = slim.stack(logits, slim.fully_connected, self.final_fc_layer_widths, scope='fc')
            logits = slim.fully_connected(
                logits, num_tasks, activation_fn=None, scope='output-fc')

            return logits
