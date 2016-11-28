from __future__ import absolute_import, division, print_function

from builtins import zip
import itertools
import numpy as np
import os
import psutil

from keras import backend as K
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten, Merge,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.recurrent import GRU
from keras.objectives import binary_crossentropy
from keras import optimizers
from keras.regularizers import l1
from keras.utils.generic_utils import Progbar

from .io_utils import generate_from_intervals, generate_from_intervals_and_labels, roundrobin
from .metrics import ClassificationResult, AMBIG_LABEL

def class_weights(y):
    """
    Parameters
    ----------
    y : 1darray
    """
    assert len(np.shape(y))==1
    total = (y >= 0).sum()
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()

    return total / num_neg, total / num_pos


def get_weighted_binary_crossentropy(w0_weights, w1_weights):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    w0_weights = np.array(w0_weights)
    w1_weights = np.array(w1_weights)
    def binary_crossentropy(y_true, y_pred): 
        weightsPerTaskRep = y_true * w1_weights[None, :] + (1 - y_true) * w0_weights[None, :]
        nonAmbig = (y_true > -0.5)
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_pred, y_true) * nonAmbigTimesWeightsPerTask, axis=-1)
    return binary_crossentropy


def build_masked_loss(loss_function, mask_value=AMBIG_LABEL):
    def binary_crossentropy(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return binary_crossentropy


def masked_binary_crossentropy(mask_value=AMBIG_LABEL):
    return build_masked_loss(binary_crossentropy, mask_value=mask_value)


class PrintMetrics(Callback):

        def __init__(self, validation_data, sequence_DNN):
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN

        def on_epoch_end(self, epoch, logs={}):
            print('\n{}\n'.format(self.sequence_DNN.test(self.X_valid, self.y_valid)))


class SequenceClassifier(object):
    def __init__(self, seq_length=None, num_tasks=None, arch_fname=None, weights_fname=None,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 pool_width=35, L1=0, dropout=0.0,
                 use_RNN=False, GRU_size=35, TDD_size=15,
                 num_epochs=100, verbose=1):
        self.saved_params = locals()
        self.verbose = verbose
        self.num_epochs = num_epochs
        if arch_fname is not None and weights_fname is not None:
            from keras.models import model_from_json
            self.model = model_from_json(open(arch_fname).read())
            self.model.load_weights(weights_fname)
            self.num_tasks = self.model.layers[-1].output_shape[-1]
        elif seq_length is not None and num_tasks is not None:
            self.num_tasks = num_tasks
            self.model = Sequential()
            assert len(num_filters) == len(conv_width)
            for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
                conv_height = 4 if i == 0 else 1
                self.model.add(Convolution2D(
                    nb_filter=nb_filter, nb_row=conv_height,
                    nb_col=nb_col, activation='linear',
                    init='he_normal', input_shape=(1, conv_height, seq_length),
                    W_regularizer=l1(L1), b_regularizer=l1(L1)))
                self.model.add(Activation('relu'))
                self.model.add(Dropout(dropout))
                #self.model.add(MaxPooling2D(pool_size=(1, pool_width[i])))
            #self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
            self.model.add(AveragePooling2D(pool_size=(1, pool_width)))
            if use_RNN:
                num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
                self.model.add(Reshape((num_filters[-1], num_max_pool_outputs)))
                self.model.add(Permute((2, 1)))
                self.model.add(GRU(GRU_size, return_sequences=True))
                self.model.add(TimeDistributedDense(TDD_size, activation='linear'))
            self.model.add(Flatten())
            self.model.add(Dense(500))
            self.model.add(Dense(output_dim=self.num_tasks))
            self.model.add(Activation('sigmoid'))
        else:
            raise RuntimeError("Model initialization requires seq_length and num_tasks or arch/weights files!")

    def compile(self, optimizer='adam', lr=0.0003, y=None):
        """
        Defines learning parameters and compiles the model.

        Parameters
        ----------
        y : 2darray, optional
           Uses class-weighted cross entropy loss if provides.
           Otherwise, uses non-weighted cross entropy loss.
        """
        if y is not None:
            task_weights = np.array([class_weights(y[:, i]) for i in range(y.shape[1])])
            loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
        else:
            loss_func = masked_binary_crossentropy()
        optimizer_cls = getattr(optimizers, optimizer)
        optimizer = optimizer_cls(lr=lr)
        self.model.compile(optimizer=optimizer, loss=loss_func)

    def train(self, X, y, validation_data,
              patience=5, verbose=True, batch_size=128, reweigh_loss_func=False):
        if reweigh_loss_func:
            task_weights = np.array([class_weights(y[:, i]) for i in range(y.shape[1])])
            loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
            self.model.compile(optimizer='adam', loss=loss_func)
        # define callbacks and fit
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        if verbose:
            self.callbacks.append(PrintMetrics(validation_data, self))
        self.model.fit(
            X, y, batch_size=batch_size, nb_epoch=self.num_epochs, shuffle=False,
            validation_data=validation_data,
            callbacks=self.callbacks, verbose=verbose)


    def deeplift(self, X, batch_size=200):
        """
        Returns (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
        """
        assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
        from deeplift import keras_conversion as kc
        from deeplift.blobs import MxtsMode
        # normalize sequence convolution weights
        kc.mean_normalise_first_conv_layer_weights(self.model, None)
        # run deeplift
        deeplift_model = kc.convert_sequential_model(
            self.model, mxts_mode=MxtsMode.DeepLIFT)
        target_contribs_func = deeplift_model.get_target_contribs_func(
            find_scores_layer_idx=0)
        return np.asarray([
            target_contribs_func(task_idx=i, input_data_list=[X],
                                 batch_size=batch_size, progress_update=None)
            for i in range(self.num_tasks)])

    def save(self, prefix):
        arch_fname = prefix + '.arch.json'
        weights_fname = prefix + '.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)[metric]

    def predict(self, X, batch_size=128):
        return self.model.predict(X, batch_size=batch_size, verbose=False)

    
class StreamingSequenceClassifier(SequenceClassifier):

    def train_on_multiple_datasets(self, dataset2train_regions_and_labels, dataset2valid_regions_and_labels, dataset2fasta_extractor,
                                   task_names=None, save_best_model_to_prefix=None,
                                   early_stopping_metric='auROC', num_epochs=100,
                                   batch_size=500, epoch_size=250000,
                                   early_stopping_patience=5, verbose=True, reweigh_loss_func=False):
        process = psutil.Process(os.getpid())
        # define training generator
        dataset2training_generator = {}
        for dataset_id, (regions, labels) in dataset2train_regions_and_labels.items():
           dataset2training_generator[dataset_id] = generate_from_intervals_and_labels(regions, labels, dataset2fasta_extractor[dataset_id],
                                                                                       batch_size=batch_size, indefinitely=True)
        training_generator = roundrobin(*dataset2training_generator.values())
        # define training loop
        valid_metrics = []
        best_metric = np.inf if early_stopping_metric == 'Loss' else -np.inf
        samples_per_epoch = len(y_train) if epoch_size is None else epoch_size
        batches_per_epoch = int(samples_per_epoch / batch_size)
        samples_per_epoch = batch_size * batches_per_epoch
        for epoch in range(1, num_epochs + 1):
            progbar = Progbar(target=samples_per_epoch)
            for batch_indxs in xrange(1, batches_per_epoch + 1):
                x, y = next(training_generator)
                batch_loss = self.model.train_on_batch(x, y)
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indxs*batch_size,
                               values=[("loss", sum(batch_loss)/len(batch_loss)), ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

            dataset2metrics, epoch_valid_metrics = self.test_on_multiple_datasets(dataset2valid_regions_and_labels, dataset2fasta_extractor,
                                                                                  task_names=task_names)
            valid_metrics.append(epoch_valid_metrics)
            if verbose:
                print('\nEpoch {}:'.format(epoch))
                for dataset_id, dataset_metrics in dataset2metrics.items():
                    print('Dataset {}:\n{}\n'.format(dataset_id, dataset_metrics), end='')
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

    def train(self, intervals_train, y_train,
              intervals_valid, y_valid, fasta_extractor,
              save_best_model_to_prefix=None,
              early_stopping_metric='auPRC', num_epochs=100,
              batch_size=128, epoch_size=100000,
              early_stopping_patience=5, verbose=True, reweigh_loss_func=False):
        process = psutil.Process(os.getpid())
        if reweigh_loss_func:
            task_weights = np.array([class_weights(y[:, i]) for i in range(y.shape[1])])
            loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
            self.model.compile(optimizer='adam', loss=loss_func)
        # define generators
        batch_array = np.zeros((batch_size, 1, 4, intervals_train[0].length), dtype=np.float32)
        training_generator = generate_from_intervals_and_labels(intervals_train, y_train, fasta_extractor,
                                                                batch_size=128, indefinitely=True, batch_array=batch_array)
        valid_metrics = []
        best_metric = np.inf if early_stopping_metric == 'Loss' else -np.inf
        samples_per_epoch = len(y_train) if epoch_size is None else epoch_size
        batches_per_epoch = int(samples_per_epoch / batch_size)
        samples_per_epoch = batch_size * batches_per_epoch # leave out leftover examples for next epoch
        for epoch in range(1, num_epochs + 1):
            progbar = Progbar(target=samples_per_epoch)
            for batch_indxs in xrange(1, batches_per_epoch + 1):
                x, y = next(training_generator)
                batch_loss = self.model.train_on_batch(x, y)
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indxs*batch_size,
                               values=[("loss", sum(batch_loss)/len(batch_loss)), ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

            epoch_valid_metrics = self.test(intervals_valid, y_valid, fasta_extractor)
            valid_metrics.append(epoch_valid_metrics)
            if verbose:
                print('\nEpoch {}:'.format(epoch))
                print('{}\n'.format(epoch_valid_metrics), end='')
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

    def predict(self, intervals, fasta_extractor, batch_size=128, verbose=False):
        """
        Generates data and returns a single 2d array with predictions.
        """
        generator = generate_from_intervals(
            intervals, fasta_extractor, indefinitely=False, batch_size=batch_size)
        if verbose:
            process = psutil.Process(os.getpid())
            progbar = Progbar(target=len(intervals))
            predictions = []
            batch_indx = 1
            for batch in generator:
                predictions.append(np.vstack(self.model.predict_on_batch(batch)))
                progbar.update(batch_indx*batch_size)
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indx*batch_size, values=[("Non-shared RSS (Mb)", rss_minus_shr_memory)])
                batch_indx += 1
        else:
            predictions = [np.vstack(self.model.predict_on_batch(batch))
                           for batch in generator]
        return np.vstack(predictions)

    def compare_predict_and_stream_predict(self, intervals, fasta_extractor, batch_size=128):
        X = fasta_extractor(intervals)
        predictions = super(StreamingSequenceClassifier, self).predict(X)
        stream_predictions = self.predict(intervals, fasta_extractor, batch_size=batch_size)
        diff = (predictions - stream_predictions).sum()
        assert diff == 0, "difference is {}".format(diff)

    def test(self, intervals, y, fasta_extractor, task_names=None):
        predictions = self.predict(intervals, fasta_extractor)
        return ClassificationResult(y, predictions, task_names=task_names)

    def test_on_multiple_datasets(self, dataset2test_regions_and_labels, dataset2fasta_extractor,
                                  batch_size=128, task_names=None, verbose=False):
        """
        Returns dctionary with dataset ids as keys and classification results as values, and a combined classification result.
        """
        dataset2classification_result = {}
        predictions_list = []
        labels_list = []
        for dataset_id, (regions, labels) in dataset2test_regions_and_labels.items():
            predictions = self.predict(regions, dataset2fasta_extractor[dataset_id], batch_size=batch_size, verbose=verbose)
            dataset2classification_result[dataset_id] = ClassificationResult(labels, predictions, task_names=task_names)
            predictions_list.append(predictions)
            labels_list.append(labels)
        predictions = np.vstack(predictions_list)
        y = np.vstack(labels_list)
        combined_classification_result = ClassificationResult(y, predictions, task_names=task_names)

        return (dataset2classification_result, combined_classification_result)

    def deeplift(self, intervals, fasta_extractor, batch_size=200):
        from deeplift import keras_conversion as kc
        from deeplift.blobs import MxtsMode
        # normalize sequence convolution weights
        kc.mean_normalise_first_conv_layer_weights(self.model, None)
        # run deeplift
        deeplift_model = kc.convert_sequential_model(
            self.model, mxts_mode=MxtsMode.DeepLIFT)
        target_contribs_func = deeplift_model.get_target_contribs_func(
                find_scores_layer_idx=0)
        generator = generate_signals_from_intervals(
            intervals, fasta_extractor, batch_size=batch_size, indefinitely=False)
        dl_scores = []
        for batch in generator:
            batch_dl_scores = np.asarray([
                target_contribs_func(task_idx=i, input_data_list=[batch],
                                 batch_size=batch_size, progress_update=None)
                for i in range(self.num_tasks)])
            dl_scores.append(batch_dl_scores)
        return np.concatenate(tuple(dl_scores), axis=1)


class StreamingSequenceAndDnaseClassifier(StreamingSequenceClassifier):

    def __init__(self, seq_length=None, num_tasks=None, arch_fname=None, weights_fname=None,
                 num_seq_filters=(25, 25, 25), seq_conv_width=(25, 25, 25),
                 num_dnase_filters=(25, 25, 25), dnase_conv_width=(25, 25, 25),
                 pool_width=25, L1=0, dropout=0.0,
                 num_epochs=100, verbose=1):
        self.saved_params = locals()
        self.verbose = verbose
        self.num_epochs = num_epochs
        if arch_fname is not None and weights_fname is not None:
            from keras.models import model_from_json
            self.model = model_from_json(open(arch_fname).read())
            self.model.load_weights(weights_fname)
            self.num_tasks = self.model.layers[-1].output_shape[-1]
        elif seq_length is not None and num_tasks is not None:
            assert len(num_seq_filters) == len(seq_conv_width)
            assert len(num_dnase_filters) == len(dnase_conv_width)

            self.num_tasks = num_tasks
            self.model = Sequential()

            def build_cnn(seq_length, num_channels, num_filters_list, conv_width_list):
                cnn = Sequential()
                cnn.add(Convolution2D(
                    num_filters_list[0],
                    num_channels, conv_width_list[0],
                    activation="relu", init="he_normal",
                    input_shape=(1, num_channels, seq_length)
                ))
                cnn.add(Dropout(dropout))
                for i in xrange(1, len(num_filters_list)):
                    cnn.add(Convolution2D(
                        num_filters_list[i],
                        1, conv_width_list[i],
                        activation="relu", init="he_normal",
                        W_regularizer=l1(L1)
                    ))
                    cnn.add(Dropout(dropout))

                return cnn

            seq_model = build_cnn(seq_length, 4, num_seq_filters, seq_conv_width)
            dnase_model = build_cnn(seq_length, 1, num_dnase_filters, dnase_conv_width)
            self.model.add(Merge([seq_model, dnase_model], mode='concat', concat_axis=2))
            self.model.add(Convolution2D( # TODO: add num_combined_filters, combined_conv_width to init 
                55, 2, 25,
                border_mode="same",
                activation="relu", init="he_normal",
                W_regularizer=l1(L1)
            ))
            self.model.add(AveragePooling2D(pool_size=(1, pool_width)))
            self.model.add(Flatten())
            self.model.add(Dense(output_dim=self.num_tasks))
            self.model.add(Activation('sigmoid'))
        else:
            raise RuntimeError("Model initialization requires seq_length and num_tasks or arch/weights files!")

    def train_on_multiple_datasets(self, dataset2train_regions_and_labels, dataset2valid_regions_and_labels, dataset2extractors,
                                   task_names=None, save_best_model_to_prefix=None,
                                   early_stopping_metric='auPRC', num_epochs=100,
                                   batch_size=32, epoch_size=250000,
                                   early_stopping_patience=5, verbose=True, reweigh_loss_func=False):
        process = psutil.Process(os.getpid())
        # define training generator
        dataset2training_generator = {}
        for dataset_id, (regions, labels) in dataset2train_regions_and_labels.items():
            dataset2training_generator[dataset_id] = generate_from_intervals_and_labels(regions, labels, dataset2extractors[dataset_id],
                                                                                        batch_size=batch_size, indefinitely=True)
        if len(dataset2training_generator) > 1:
            def concatenate_training_generators(generators):
                for batches in zip(*generators):
                    sequence = np.concatenate([batch[0][0] for batch in batches], axis=0)
                    dnase = np.concatenate([batch[0][1] for batch in batches], axis=0)
                    y = np.concatenate([batch[1] for batch in batches], axis=0)
                    yield ([sequence, dnase], y)
            training_generator = concatenate_training_generators(dataset2training_generator.values())
            batch_size *= len(dataset2training_generator)
        else:
            training_generator = roundrobin(*dataset2training_generator.values())
        # define training loop
        valid_metrics = []
        print("Getting initial validation metrics..")
        _, epoch_valid_metrics = self.test_on_multiple_datasets(dataset2valid_regions_and_labels, dataset2extractors,
                                                                              task_names=task_names)
        valid_metrics.append(epoch_valid_metrics)
        best_metric = epoch_valid_metrics[early_stopping_metric].mean()
        print('Initial {}: {:.3f}'.format(early_stopping_metric, best_metric))
        samples_per_epoch = len(y_train) if epoch_size is None else epoch_size
        batches_per_epoch = int(samples_per_epoch / batch_size)
        samples_per_epoch = batch_size * batches_per_epoch
        for epoch in range(1, num_epochs + 1):
            progbar = Progbar(target=samples_per_epoch)
            for batch_indxs in xrange(1, batches_per_epoch + 1):
                x, y = next(training_generator)
                batch_loss = self.model.train_on_batch(x, y)
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indxs*batch_size,
                               values=[("loss", sum(batch_loss)/len(batch_loss)), ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

            dataset2metrics, epoch_valid_metrics = self.test_on_multiple_datasets(dataset2valid_regions_and_labels, dataset2extractors,
                                                                                  task_names=task_names)
            valid_metrics.append(epoch_valid_metrics)
            if verbose:
                print('\nEpoch {}:'.format(epoch))
                for dataset_id, dataset_metrics in dataset2metrics.items():
                    print('Dataset {}:\n{}\n'.format(dataset_id, dataset_metrics), end='')
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
