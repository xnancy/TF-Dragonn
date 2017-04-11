from __future__ import absolute_import, division, print_function

from builtins import zip
from genomedatalayer.extractors import (
    FastaExtractor, MemmappedBigwigExtractor, MemmappedFastaExtractor
)
import itertools
import numpy as np


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


def generate_from_intervals(intervals, extractors, batch_size=128, indefinitely=True):
    """
    Generates signals extracted on interval batches.

    Parameters
    ----------
    intervals : sequence of intervals
    extractors : list of gdl extractors
    batch_size : int, optional
    indefinitely : bool, default: True
    """
    interval_length = intervals[0].length
    # preallocate batch arrays
    batch_arrays = []
    for extractor in extractors:
        if type(extractor) in [FastaExtractor, MemmappedFastaExtractor]:
            batch_arrays.append(
                np.zeros((batch_size, 1, 4, interval_length), dtype=np.float32))
        elif type(extractor) in [MemmappedBigwigExtractor]:
            batch_arrays.append(
                np.zeros((batch_size, 1, 1, interval_length), dtype=np.float32))
    if indefinitely:
        batch_iterator = infinite_batch_iter(intervals, batch_size)
    else:
        batch_iterator = batch_iter(intervals, batch_size)
    for batch_intervals in batch_iterator:
        try:
            yield [extractor(batch_intervals, out=batch_arrays[i]) for i, extractor in enumerate(extractors)]
        except ValueError:
            yield [extractor(batch_intervals) for extractor in extractors]


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def test_extractor_in_generator(intervals, extractor, batch_size=128):
    """
    Extracts data in bulk, then in streaming batches and checks its the same data.
    """
    from keras.utils.generic_utils import Progbar

    X_in_memory = extractor(intervals)
    samples_per_epoch = len(intervals)
    batches_per_epoch = int(samples_per_epoch / batch_size) + 1
    batch_array = np.zeros(
        (batch_size, 1, 4, intervals[0].length), dtype=np.float32)
    batch_generator = generate_from_intervals(
        intervals, extractor, batch_size=batch_size, indefinitely=False, batch_array=batch_array)
    progbar = Progbar(target=samples_per_epoch)
    for batch_indx in xrange(1, batches_per_epoch + 1):
        X_batch = next(batch_generator)
        start = (batch_indx - 1) * batch_size
        stop = batch_indx * batch_size
        if stop > samples_per_epoch:
            stop = samples_per_epoch
        # assert streamed sequences and labels match data in memory
        assert (X_in_memory[start:stop] - X_batch).sum() == 0
        progbar.update(stop)


def generate_from_array(array, batch_size=128, indefinitely=True):
    """
    Generates the array in batches.
    """
    if indefinitely:
        batch_iterator = infinite_batch_iter(array, batch_size)
    else:
        batch_iterator = batch_iter(array, batch_size)
    for array_batch in batch_iterator:
        yield np.stack(array_batch, axis=0)


def generate_from_intervals_and_labels(intervals, labels, extractors, batch_size=128, indefinitely=True):
    """
    Generates batches of (inputs, labels) where inputs is a list of numpy arrays based on provided extractors.
    """
    batch_generator = zip(generate_from_intervals(intervals, extractors,
                                                  batch_size=batch_size,
                                                  indefinitely=indefinitely),
                          generate_from_array(labels, batch_size=batch_size, indefinitely=indefinitely))
    for batch in batch_generator:
        yield batch
