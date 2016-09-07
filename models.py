from __future__ import absolute_import, division, print_function
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

from dragonn.models import SequenceDNN
from metrics import ClassificationResult


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
    def weighted_binary_crossentropy(y_true, y_pred): 
        weightsPerTaskRep = y_true * w1_weights[None, :] + (1 - y_true) * w0_weights[None, :]
        nonAmbig = (y_true > -0.5)
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_pred, y_true) * nonAmbigTimesWeightsPerTask, axis=-1)
    return weighted_binary_crossentropy


class PrintMetrics(Callback):

        def __init__(self, validation_data, sequence_DNN):
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN

        def on_epoch_end(self, epoch, logs={}):
            print('\n{}\n'.format(self.sequence_DNN.test(self.X_valid, self.y_valid)))


class SequenceClassifier(SequenceDNN):
    def __init__(self, X_train, y_train, use_RNN=False,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 pool_width=35, GRU_size=35, TDD_size=15,
                 L1=0, dropout=0.0, num_epochs=100, verbose=1):
        self.saved_params = locals()
        self.seq_length = X_train.shape[-1]
        self.input_shape = X_train.shape[1:]
        self.num_tasks = y_train.shape[1]
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.model = Sequential()
        assert len(num_filters) == len(conv_width)
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            conv_height = 4 if i == 0 else 1
            self.model.add(Convolution2D(
                nb_filter=nb_filter, nb_row=conv_height,
                nb_col=nb_col, activation='linear',
                init='he_normal', input_shape=self.input_shape,
                W_regularizer=l1(L1), b_regularizer=l1(L1)))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout))
        self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(Flatten())
        self.model.add(Dense(output_dim=self.num_tasks))
        self.model.add(Activation('sigmoid'))
        # get task specific class weights
        task_weights = np.array([class_weights(y_train[:, i]) for i in range(self.num_tasks)])
        # get weighted cross entropy loss and compile
        loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
        self.model.compile(optimizer='adam', loss=loss_func)

    def train(self, X, y, validation_data, patience=5, verbose=True, batch_size=128):
        # define callbacks and fit
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        if verbose:
            self.callbacks.append(PrintMetrics(validation_data, self))
        self.model.fit(
            X, y, batch_size=batch_size, nb_epoch=self.num_epochs,
            validation_data=validation_data,
            callbacks=self.callbacks, verbose=verbose)

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)[metric]
        
    
