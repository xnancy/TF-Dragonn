from __future__ import absolute_import, division, print_function
import itertools
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l1

from .metrics import ClassificationResult

def batch_iter(iterable, batch_size):
    '''iterates in batches.
    '''
    it = iter(iterable)
    try:
        while True:
            values = []
            for n in xrange(batch_size):
                values += (it.next(),)
            yield values
    except StopIteration:
        # yield remaining values
        yield values


def infinite_batch_iter(iterable, batch_size):
    '''iterates in batches indefinitely.
    '''
    return batch_iter(itertools.cycle(iterable),
                      batch_size)


def generate_signals_from_intervals(intervals, extractor, batch_size=128):
    """
    Generates signals extracted on interval batches.
    """
    batch_iterator = infinite_batch_iter(intervals, batch_size)
    for batch_intervals in batch_iterator:
        yield extractor(batch_intervals)


def generate_array_batches(array, batch_size=128):
    """
    Generates the array in batches.
    """
    for array_batch in infinite_batch_iter(array, batch_size):
        yield np.vstack(array_batch)


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
                conv_height = X.shape[-2] if i == 0 else 1
                self.model.add(Convolution2D(
                    nb_filter=nb_filter, nb_row=conv_height,
                    nb_col=nb_col, activation='linear',
                    init='he_normal', input_shape=(1, 4, seq_length),
                    W_regularizer=l1(L1), b_regularizer=l1(L1)))
                self.model.add(Activation('relu'))
                self.model.add(Dropout(dropout))
            self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
            if use_RNN:
                num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
                self.model.add(Reshape((num_filters[-1], num_max_pool_outputs)))
                self.model.add(Permute((2, 1)))
                self.model.add(GRU(GRU_size, return_sequences=True))
                self.model.add(TimeDistributedDense(TDD_size, activation='linear'))
            self.model.add(Flatten())
            self.model.add(Dense(output_dim=self.num_tasks))
            self.model.add(Activation('sigmoid'))
            # get task specific class weights
            task_weights = np.array([class_weights(y[:, i]) for i in range(self.num_tasks)])
            # get weighted cross entropy loss and compile
            loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
            self.model.compile(optimizer='adam', loss=loss_func)
        else:
            raise RuntimeError("Model initialization requires seq_length and num_tasks or arch/weights files!")

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
            X, y, batch_size=batch_size, nb_epoch=self.num_epochs,
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

    def train(self, intervals_train, y_train,
              intervals_valid, y_valid, fasta_extractor,
              patience=5, verbose=True, batch_size=128, reweigh_loss_func=False):
        if reweigh_loss_func:
            task_weights = np.array([class_weights(y[:, i]) for i in range(y.shape[1])])
            loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
            self.model.compile(optimizer='adam', loss=loss_func)
        # define callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        if verbose:
            self.callbacks.append(PrintMetrics(validation_data, self))
        # define generators
        training_generator = zip(generate_signals_from_intervals(intervals_train, fasta_extractor),
                                 generate_array_batches(y_train, batch_size=batch_size))
        validation_generator = zip(generate_signals_from_intervals(intervals_valid, fasta_extractor),
                                   generate_array_batches(labels_valid, batch_size=batch_size))
        self.model.fit_generator(training_generator, 50000, 4,
                                 validation_data=validation_generator,
                                 nb_val_samples=len(intervals_valid))

    def predict(self, intervals, fasta_extractor, batch_size=128):
        generator = generate_signals_from_intervals(intervals, fasta_extractor)
        return self.model.predict_generator(self, generator, len(intervals))
