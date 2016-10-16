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


def generate_signals_from_intervals(intervals, extractor, batch_size=128, indefinitely=True, batch_array=None):
    """
    Generates signals extracted on interval batches.
    """
    if batch_array is not None:
        batch_size = len(batch_array)
    else:
        interval_length = intervals[0].length
        batch_array = np.zeros((batch_size, 1, 4, interval_length), dtype=np.float32)
    if indefinitely:
        batch_iterator = infinite_batch_iter(intervals, batch_size)
    else:
        batch_iterator = batch_iter(intervals, batch_size)
    for batch_intervals in batch_iterator:
        try:
            yield extractor(batch_intervals, out=batch_array)
        except ValueError:
            yield extractor(batch_intervals)


def generate_array_batches(array, batch_size=128, indefinitely=True):
    """
    Generates the array in batches.
    """
    if indefinitely:
        batch_iterator = infinite_batch_iter(array, batch_size)
    else:
        batch_iterator = batch_iter(array, batch_size)
    for array_batch in batch_iterator:
        yield np.vstack(array_batch)


def generate_from_intervals_and_labels(intervals, labels, extractor, batch_size=128, indefinitely=True, batch_array=None):
    batch_generator = zip(generate_signals_from_intervals(intervals, extractor,
                                                          batch_array=batch_array,
                                                          batch_size=batch_size,
                                                          indefinitely=indefinitely),
                          generate_array_batches(labels, batch_size=batch_size, indefinitely=indefinitely))
    for batch in batch_generator:
        yield batch
