from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty, ABCMeta
import tensorflow as tf
import tensorflow.contrib.slim as slim

import initializers


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

    @property
    def num_tasks(self):
        return self._num_tasks


class SequenceAndDnaseClassifier(Classifier):

    @property
    def get_inputs(self):
        return ["data/genome_data_dir", "data/dnase_data_dir"]

    def __init__(self, num_tasks=1,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 num_combined_filters=(55,), combined_conv_width=(25,),
                 pool_width=25, batch_norm=False):
        assert len(num_seq_filters) == len(seq_conv_width)
        assert len(num_dnase_filters) == len(dnase_conv_width)
        assert len(num_combined_filters) == len(combined_conv_width)

        self._num_tasks = num_tasks
        self.num_seq_filters = num_seq_filters
        self.seq_conv_width = seq_conv_width
        self.num_dnase_filters = num_dnase_filters
        self.dnase_conv_width = dnase_conv_width
        self.num_combined_filters = num_combined_filters
        self.combined_conv_width = combined_conv_width
        self.pool_width = pool_width
        self.batch_norm = batch_norm

    def get_logits(self, inputs):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected], reuse=False, activation_fn=tf.nn.relu,
                weights_initializer=initializers.he_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0)):

            def expand_4D(input_tensor):
                shape = [x.value for x in input_tensor.get_shape()]
                if len(shape) == 2:  # 1-D input
                    new_shape = [shape[0], 1, shape[1], 1]
                elif len(shape) == 3:  # 2-D input
                    new_shape = shape + [1]
                else:
                    raise IOError('unrecognized shape: {}'.format(shape))
                return tf.reshape(input_tensor, new_shape)

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
            logits = tf.concat(1, [seq_preds, dnase_preds])
            print('concat shape {}'.format(logits.get_shape()))
            for i, (num_filter, filter_width) in enumerate(
                    zip(self.num_combined_filters, self.combined_conv_width)):
                filter_height = 2 if i == 0 else 1
                filter_dims = [filter_height, filter_width]
                logits = slim.conv2d(logits, num_filter, filter_dims, padding='VALID',
                                     scope='combined_conv{:d}'.format(i + 1))
            print('after combined conv layers shape {}'.format(logits.get_shape()))
            logits = slim.avg_pool2d(logits, [1, self.pool_width], stride=[1, self.pool_width],
                                     padding='VALID', scope='avg_pool')
            logits = slim.flatten(logits, scope='flatten')
            logits = slim.fully_connected(
                logits, self.num_tasks, activation_fn=None, scope='fc')

            return logits
