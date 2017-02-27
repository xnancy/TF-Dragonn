from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import psutil
import six.moves

from keras import backend as K, optimizers
from keras.objectives import binary_crossentropy
from keras.utils.generic_utils import Progbar

from metrics import ClassificationResult, AMBIG_LABEL
import io_utils

BATCH_FREQ_UPDATE_MEM_USAGE = 100
BATCH_FREQ_UPDATE_PROGBAR = 50


def build_masked_loss(loss_function, mask_value=AMBIG_LABEL):
    def binary_crossentropy(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return binary_crossentropy


def masked_binary_crossentropy(mask_value=AMBIG_LABEL):
    return build_masked_loss(binary_crossentropy, mask_value=mask_value)


class ClassifierTrainer(object):

    def __init__(self, optimizer='adam', lr=0.0003, batch_size=128,
                 epoch_size=250000, num_epochs=100,
                 early_stopping_metric='auPRC', early_stopping_patience=5,
                 task_names=None, logger=None):
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_epochs = num_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.task_names = task_names
        self.logger = logger

    def compile(self, model):
        loss_func = masked_binary_crossentropy()
        optimizer_cls = getattr(optimizers, self.optimizer)
        optimizer = optimizer_cls(lr=self.lr)
        model.model.compile(optimizer=optimizer, loss=loss_func)

    def train(self, model, train_queue, valid_queue,
              save_best_model_to_prefix=None, verbose=True):
        self.logger.info('optimizer: {}'.format(self.optimizer))
        self.logger.info('learning rate: {}'.format(self.lr))
        self.logger.info('batch size: {}'.format(self.batch_size))
        self.logger.info('epoch size: {}'.format(self.epoch_size))
        self.logger.info('max num of epochs: {}'.format(self.num_epochs))
        self.logger.info('early stopping metrics: {}'.format(self.early_stopping_metric))
        self.logger.info('early stopping patience: {}'.format(self.early_stopping_patience))
        process = psutil.Process(os.getpid())

        self.compile(model)

        def get_rss_prop():  # this is quite expensive
            return (process.memory_info().rss - process.memory_info().shared) / 10**6

        train_iterator = None
        try:
            train_iterator = io_utils.ExampleQueueIterator(
                train_queue, num_exs_batch=self.batch_size,
                num_epochs=self.num_epochs, num_exs_epoch=self.epoch_size)

            valid_metrics = []
            best_metric = np.inf if self.early_stopping_metric == 'Loss' else -np.inf
            batches_per_epoch = int(
                np.floor(self.epoch_size / self.batch_size))
            samples_per_epoch = self.batch_size * batches_per_epoch

            for epoch in six.moves.range(1, self.num_epochs + 1):
                progbar = Progbar(target=samples_per_epoch)
                rss_minus_shr_memory = get_rss_prop()

                for batch_indxs in six.moves.range(1, batches_per_epoch + 1):
                    batch = train_iterator.next()
                    batch_loss = model.model.train_on_batch(
                        batch, batch['labels'])

                    if batch_indxs % BATCH_FREQ_UPDATE_MEM_USAGE == 0:
                        rss_minus_shr_memory = get_rss_prop()

                    if batch_indxs % BATCH_FREQ_UPDATE_PROGBAR == 0:
                        progbar.update(batch_indxs * self.batch_size,
                                       values=[("loss", batch_loss),
                                               ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

                epoch_valid_metrics = self.test(model, valid_queue, test_size=500000)
                valid_metrics.append(epoch_valid_metrics)
                if verbose:
                    self.logger.info('\nEpoch {}:'.format(epoch))
                    self.logger.info('Metrics across all datasets:\n{}\n'.format(
                        epoch_valid_metrics))
                current_metric = epoch_valid_metrics[
                    self.early_stopping_metric].mean()
                if (self.early_stopping_metric == 'Loss') == (current_metric <= best_metric):
                    if verbose:
                        self.logger.info('New best {}. Saving model.\n'.format(
                            self.early_stopping_metric))
                    best_metric = current_metric
                    best_epoch = epoch
                    early_stopping_wait = 0
                    if save_best_model_to_prefix is not None:
                        model.save(save_best_model_to_prefix)
                else:
                    if early_stopping_wait >= self.early_stopping_patience:
                        break
                    early_stopping_wait += 1
            train_iterator.close()

        except Exception as e:
            if train_iterator is not None:
                train_iterator.close()
            raise e

        if verbose:  # end of training messages
            self.logger.info('Finished training after {} epochs.'.format(epoch))
            if save_best_model_to_prefix is not None:
                self.logger.info("The best model's architecture and weights (from epoch {0}) "
                                 'were saved to {1}.arch.json and {1}.weights.h5'.format(
                                     best_epoch, save_best_model_to_prefix))

    def test(self, model, queue, batch_size=1000, verbose=True, test_size=None):
        iterator = None
        process = psutil.Process(os.getpid())

        def get_rss_prop():  # this is quite expensive
            return (process.memory_info().rss - process.memory_info().shared) / 10**6
        rss_minus_shr_memory = get_rss_prop()

        try:
            iterator = io_utils.ExampleQueueIterator(
                queue, num_exs_batch=batch_size, num_epochs=1,
                allow_smaller_final_batch=True)
            if test_size is not None:
                num_examples = min(test_size, iterator.num_examples)
            else:
                num_examples = iterator.num_examples
            num_batches = int(np.floor(iterator.num_examples / batch_size))

            if verbose:
                progbar = Progbar(target=num_examples)

            predictions = []
            labels = []

            for batch_indx, batch in enumerate(iterator):
                predictions.append(
                    np.vstack(model.model.predict_on_batch(batch)))
                labels.append(batch['labels'])
                if verbose:
                    if batch_indx % BATCH_FREQ_UPDATE_MEM_USAGE == 0:
                        rss_minus_shr_memory = get_rss_prop()
                    if batch_indx % BATCH_FREQ_UPDATE_PROGBAR == 0:
                        progbar.update(batch_indx * batch_size,
                                       values=[("Non-shared RSS (Mb)", rss_minus_shr_memory)])
            iterator.close()
            del iterator

        except Exception as e:
            if iterator is not None:
                iterator.close()
            raise e

        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        return ClassificationResult(labels, predictions, task_names=self.task_names)
