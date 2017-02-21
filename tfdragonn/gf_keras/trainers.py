from __future__ import absolute_import, division, print_function

from builtins import zip
import numpy as np
import os
import psutil

from keras import backend as K, optimizers
from keras.objectives import binary_crossentropy
from keras.utils.generic_utils import Progbar

from metrics import ClassificationResult, AMBIG_LABEL
import io_utils

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
                 task_names=None):
        self.optimizer = optimizer
        self.lr= lr
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_epochs = num_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.task_names = task_names

    def compile(self, model):
        loss_func = masked_binary_crossentropy()
        optimizer_cls = getattr(optimizers, self.optimizer)
        optimizer = optimizer_cls(lr=self.lr)
        model.model.compile(optimizer=optimizer, loss=loss_func)

    def train(self, model, train_queue, valid_queue,
              save_best_model_to_prefix=None, verbose=True):
        process = psutil.Process(os.getpid())

        self.compile(model)

        iterator_num_epochs = int(self.epoch_size / train_queue.num_examples * self.num_epochs) + 1
        train_iterator = io_utils.ExampleQueueIterator(train_queue, batch_size=self.batch_size, num_epochs=iterator_num_epochs)
        valid_metrics = []
        best_metric = np.inf if self.early_stopping_metric == 'Loss' else -np.inf
        batches_per_epoch = int(self.epoch_size / self.batch_size)
        samples_per_epoch = self.batch_size * batches_per_epoch
        for epoch in range(1, self.num_epochs + 1):
            progbar = Progbar(target=samples_per_epoch)
            for batch_indxs in xrange(1, batches_per_epoch + 1):
                batch = train_iterator.next()
                batch_loss = model.model.train_on_batch(batch, batch['labels'])
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indxs * self.batch_size,
                               values=[("loss", batch_loss), ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

            epoch_valid_metrics = self.test(model, valid_queue)
            valid_metrics.append(epoch_valid_metrics)
            if verbose:
                print('\nEpoch {}:'.format(epoch))
                print('Metrics across all datasets:\n{}\n'.format(epoch_valid_metrics), end='')
            current_metric = epoch_valid_metrics[self.early_stopping_metric].mean()
            if (self.early_stopping_metric == 'Loss') == (current_metric <= best_metric):
                if verbose:
                    print('New best {}. Saving model.\n'.format(self.early_stopping_metric))
                best_metric = current_metric
                best_epoch = epoch
                early_stopping_wait = 0
                if save_best_model_to_prefix is not None:
                    model.save(save_best_model_to_prefix)
            else:
                if early_stopping_wait >= self.early_stopping_patience:
                    break
                early_stopping_wait += 1
        if verbose: # end of training messages
            print('Finished training after {} epochs.'.format(epoch))
            if save_best_model_to_prefix is not None:
                print("The best model's architecture and weights (from epoch {0}) "
                      'were saved to {1}.arch.json and {1}.weights.h5'.format(
                    best_epoch, save_best_model_to_prefix))

    def test(self, model, queue, verbose=True):
        iterator = io_utils.ExampleQueueIterator(queue, batch_size=self.batch_size, num_epochs=1)
        num_batches = int(iterator.num_examples / self.batch_size)
        num_samples = self.batch_size * num_batches
        if verbose:
            process = psutil.Process(os.getpid())
            progbar = Progbar(target=num_samples)
        predictions = []
        labels = []
        for batch_indx in range(1, num_batches + 1):
            batch = iterator.next()
            predictions.append(np.vstack(model.model.predict_on_batch(batch)))
            labels.append(batch['labels'])
            if verbose:
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indx * self.batch_size, values=[("Non-shared RSS (Mb)", rss_minus_shr_memory)])
        del iterator

        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        return ClassificationResult(labels, predictions, task_names=self.task_names)
