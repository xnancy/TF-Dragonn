from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import re

import bcolz
import numpy as np
import tensorflow as tf

_data_cache = {}  # used to cache datasets to prevent re-loading


def bcolz_interval_reader(intervals, data_directory, interval_size=1000, norm_params=None,
                          interval_params=None, in_memory=True, op_name='bcolz-reader', use_cache=True):
    """Op to read intervals from a data_directory.

    Params:
        intervals: a dict of tensors. Either:
            1) intervals encoded as 'chrom': (string), 'start': (int), and 'end': (int), or
            2) intervals encoded as 'bed3': (string) TSV entries
        data_directory: the data directory containing chromosome subdirectories
        in_memory: whether to pre-load compressed data into memory (must be false if v-plot input)
        op_name: name given to tensorflow py_func op
        use_cache: (optional) bool of whether to used cached datasets
    Returns:
        read_values: a tf.Tensor with shape NxW of the read outputs.
    """
    with tf.variable_scope(op_name):
        if ('bed3' in intervals.keys()):
            bed3_entries_tensors = [intervals['bed3']]
        else:
            for k in ['chrom', 'start', 'end']:
                if k not in intervals.keys():
                    raise IOError(
                        'BED-3 entries string tensor must have 1 dimension.')
                bed3_entries_tensors = []
                for k in ['chrom', 'start', 'end']:
                    bed3_entries_tensors.append(intervals[k])

        if interval_params:
            with tf.variable_scope('interval_slicer'):
                assert (interval_params == "midpoint")
                bed3_entries_tensors[1] = tf.floor((bed3_entries_tensors[1]+bed3_entries_tensors[2])/2)
                bed3_entries_tensors[2] = bed3_entries_tensors[1] + 1
            interval_size = 1

        # Check what the size of the batches to read is
        read_batch_size = bed3_entries_tensors[0].get_shape()[0].value

        data = _load_directory(data_directory, in_memory, use_cache)
        data_shape = list(data[list(data.keys())[0]].shape)

        bcolz_reader_fn = _get_bcolz_reader_fn(
            data, in_memory, interval_size)
        read_values = _get_pyfunc_from_reader_fn(
            bcolz_reader_fn, bed3_entries_tensors, op_name)

        output_shape = [read_batch_size] + data_shape[:-1] + [interval_size]
        read_values = tf.reshape(read_values, output_shape)

        if norm_params:  # TODO(cprobert): implement normalization parameters
            with tf.variable_scope('normalizer'):
                assert (norm_params == 'local_zscore')
                mean, var = tf.nn.moments(read_values, [1], keep_dims=True)
                std = tf.maximum(tf.abs(tf.sqrt(var)), 0.0001)
                read_values = (read_values - mean) / std
                read_values = tf.verify_tensor_all_finite(
                    read_values, "non-finite values!")

        return read_values


def _get_pyfunc_from_reader_fn(reader_fn, bed3_entries_tensors, op_name):
    return tf.py_func(reader_fn, bed3_entries_tensors, tf.float32,
                      name=op_name)


def _get_bcolz_reader_fn(data, in_memory, interval_size):
    """Generate a function to read from a bcolz-compressed data directory.

    Params:
        bed3_entries_tensor: a tensor of BED-3 entry strings to read
        data: the data from _load_directory
        in_memory: whether to pre-load compressed data into memory (must be false if v-plot input)
    Returns:
        function that takes np.array of BED-3 strings to read
    """
    def accessor_func_1d(chrom, start, end):
        return data[chrom][start:end]

    def accessor_func_2d(chrom, start, end):
        return data[chrom][:, start:end]

    # the first dim will vary by chrom
    data_shape = list(data[list(data.keys())[0]].shape)

    if len(data_shape) == 1:
        accessor_func = accessor_func_1d
    else:
        accessor_func = accessor_func_2d

    def extractor_func(*bed3_entries):
        if bed3_entries[0].ndim != 1:
            raise IOError(
                'BED-3 tensor has wrong number of dimensions (expected 1): {}'.format(
                    bed3_entries.ndim))
        n_entries = bed3_entries[0].shape[0]

        if len(bed3_entries) == 1:  # a bed-3 tsv string tensor
            str_entries = bed3_entries.astype(str).tolist()
            chrs = []
            starts = []
            ends = []
            for str_entry in str_entries:
                split_entry = str_entry.strip().split('\t')
                chrs.append(split_entry[0])
                starts.append(int(split_entry[1]))
                ends.append(int(split_entry[2]))

        else:  # seperate chrom (str), start (int), end (int) tensors
            assert(len(bed3_entries) == 3)
            chrs = bed3_entries[0]
            starts = bed3_entries[1]
            ends = bed3_entries[2]

        output_shape = [n_entries] + data_shape[:-1] + [interval_size]
        output = np.empty(shape=output_shape, dtype=np.float32)
        for i, (chrom, start, end) in enumerate(zip(chrs, starts, ends)):
            fetched = accessor_func(chrom, start, end)
            output[i] = fetched

        return output

    return extractor_func


def _load_directory(base_dir, in_memory, use_cache):
    """Load a bcolz genome-wide data directory.

    Params:
        base_dir: string, the directory to load
        in_memory: bool, whether to copy the data to memory (not valid for v-plots)
        use_cache: bool, whether to use the global shared dataset cache
    """
    global _data_cache

    if not os.path.isdir(base_dir):
        raise IOError(
            'Base directory must be a directory: {}'.format(base_dir))

    if use_cache:
        # We need to make sure the path is correct, e.g. if trailing slashes
        # are/aren't included
        esc_sep = re.escape(os.sep)
        base_dir_key = re.sub(
            '{}{}+'.format(esc_sep, esc_sep), os.sep, base_dir)
        base_dir_key = base_dir_key.rstrip(os.sep)

        if base_dir_key in _data_cache:  # cache hit
            return _data_cache[base_dir_key]

    with open(os.path.join(base_dir, 'metadata.json'), 'r') as fp:
        metadata = json.load(fp)

    if metadata['type'] == 'array_bcolz':
        data = {chrom: bcolz.open(os.path.join(base_dir, chrom), mode='r')
                for chrom in metadata['file_shapes']}

        for chrom, shape in metadata['file_shapes'].items():
            if data[chrom].shape != tuple(shape):
                raise ValueError('Inconsistent shape found in metadata file: '
                                 '{} - {} vs {}'.format(chrom, shape,
                                                        data[chrom].shape))
    elif metadata['type'] in {'vplot_bcolz', 'array_2D_transpose_bcolz'}:
        data = {chrom: Array2D(os.path.join(base_dir, chrom), mode='r')
                for chrom in next(os.walk(base_dir))[1]}
        for chrom, shape in metadata['file_shapes'].iteritems():
            if data[chrom].shape != tuple(shape):
                raise ValueError('Inconsistent shape found in metadata file: '
                                 '{} - {} vs {}'.format(chrom, shape,
                                                        data[chrom].shape))
    else:
        raise IOError('Only bcolz arrays are supported.')

    if in_memory:
        data = {k: data[k].copy() for k in data.keys()}

    # cache the data here, so in-memory copies are kept if loaded
    if use_cache:
        _data_cache[base_dir_key] = data

    return data


class Array2D(object):
    """Representation for 2D arrays (we want row-major storage)."""

    def __init__(self, rootdir, mode='r'):
        self._arr = bcolz.open(rootdir, mode=mode)

    def __getitem__(self, key):
        r, c = key
        return self._arr[c, r].T

    def __setitem__(self, key, item):
        r, c = key
        self._arr[c, r] = item

    @property
    def shape(self):
        return self._arr.shape[::-1]

    @property
    def ndim(self):
        return self._arr.ndim
    
    def copy(self):
        self._arr = self._arr.copy()
        return self
