from __future__ import absolute_import, division, print_function

from builtins import zip
import json
import numpy as np
import os
import psutil
import sys

from keras import backend as K, optimizers
from keras.layers import Convolution1D, Input, MaxPooling1D, AveragePooling1D
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten, Permute
)
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.utils.generic_utils import Progbar

from metrics import ClassificationResult, AMBIG_LABEL
import io_utils

def model_from_config(model_config_file_path):
    """Load a model from a json config file."""
    thismodule = sys.modules[__name__]
    with open(model_config_file_path, 'r') as fp:
        config = json.load(fp)
    model_class_name = config['model_class']

    model_class = getattr(thismodule, model_class_name)
    del config['model_class']
    return model_class(**config)


def build_masked_loss(loss_function, mask_value=AMBIG_LABEL):
    def binary_crossentropy(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return binary_crossentropy


def masked_binary_crossentropy(mask_value=AMBIG_LABEL):
    return build_masked_loss(binary_crossentropy, mask_value=mask_value)


class SequenceClassifier(object):

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

    def compile(self, optimizer='adam', lr=0.0003):
        loss_func = masked_binary_crossentropy()
        optimizer_cls = getattr(optimizers, optimizer)
        optimizer = optimizer_cls(lr=lr)
        self.model.compile(optimizer=optimizer, loss=loss_func)

    def save(self, prefix):
        arch_fname = prefix + '.arch.json'
        weights_fname = prefix + '.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)

    def score(self, iterator, metric):
        return self.test(iterator)[metric]

    def train(self, train_queue, valid_queue,
              task_names=None, save_best_model_to_prefix=None,
              early_stopping_metric='auPRC',
              batch_size=128,
              epoch_size=250000,
              num_epochs=100,
              early_stopping_patience=5, verbose=True):
        process = psutil.Process(os.getpid())

        train_iterator = io_utils.ExampleQueueIterator(train_queue, batch_size=batch_size, num_epochs=1)
        batch_size = train_iterator.batch_size
        valid_metrics = []
        best_metric = np.inf if early_stopping_metric == 'Loss' else -np.inf
        batches_per_epoch = int(epoch_size / batch_size)
        samples_per_epoch = batch_size * batches_per_epoch
        for epoch in range(1, num_epochs + 1):
            progbar = Progbar(target=samples_per_epoch)
            for batch_indxs in xrange(1, batches_per_epoch + 1):
                batch = next(train_iterator)
                batch_loss = self.model.train_on_batch(batch['data/genome_data_dir'], batch['labels'])
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indxs * batch_size,
                               values=[("loss", batch_loss), ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

            epoch_valid_metrics = self.test(valid_queue, task_names=task_names)
            valid_metrics.append(epoch_valid_metrics)
            if verbose:
                print('\nEpoch {}:'.format(epoch))
                print('Metrics across all datasets:\n{}\n'.format(epoch_valid_metrics), end='')
            current_metric = epoch_valid_metrics[early_stopping_metric].mean()
            if (early_stopping_metric == 'Loss') == (current_metric <= best_metric):
                if verbose:
                    print('New best {}. Saving model.\n'.format(early_stopping_metric))
                best_metric = current_metric
                best_epoch = epoch
                early_stopping_wait = 0
                if save_best_model_to_prefix is not None:
                    self.save(save_best_model_to_prefix)
            else:
                if early_stopping_wait >= early_stopping_patience:
                    break
                early_stopping_wait += 1
        if verbose: # end of training messages
            print('Finished training after {} epochs.'.format(epoch))
            if save_best_model_to_prefix is not None:
                print("The best model's architecture and weights (from epoch {0}) "
                      'were saved to {1}.arch.json and {1}.weights.h5'.format(
                    best_epoch, save_best_model_to_prefix))

    def test(self, queue, batch_size=128, task_names=None, verbose=True):
        iterator = io_utils.ExampleQueueIterator(queue, batch_size=batch_size, num_epochs=1)
        batch_size = iterator.batch_size
        num_batches = int(iterator.num_examples / batch_size) # - 1
        num_samples = batch_size * num_batches
        if verbose:
            process = psutil.Process(os.getpid())
            progbar = Progbar(target=num_samples)
        predictions = []
        labels = []
        for batch_indx in range(1, num_batches + 1):
            batch = iterator.next()
            predictions.append(np.vstack(self.model.predict_on_batch(batch['data/genome_data_dir'])))
            labels.append(batch['labels'])
            if verbose:
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indx * batch_size, values=[("Non-shared RSS (Mb)", rss_minus_shr_memory)])

        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        return ClassificationResult(labels, predictions, task_names=task_names)
