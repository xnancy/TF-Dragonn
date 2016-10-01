from collections import OrderedDict
from itertools import groupby
from itertools import izip
import logging
import os
import subprocess

import numpy as np
from scipy.io import loadmat

from pybedtools import BedTool
from pybedtools import Interval
from pysam import tabix_index


import pyximport
_importers = pyximport.install()
from qcatIO import write_score_for_single_interval
pyximport.uninstall(*_importers)

# Setup logging
log_formatter = \
    logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')

logger = logging.getLogger('make_tracks')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# Magic numbers and constants here (that are not in args)
BLOB_DIMS = 4

def expand_dims_blob(arr, target_num_axes=BLOB_DIMS):
    # Reshapes arr, adds dims after the first axis
    assert len(arr.shape) <= BLOB_DIMS
    extra_dims = target_num_axes - len(arr.shape)
    new_shape = (arr.shape[0],) + (1,)*extra_dims + tuple(arr.shape[1:])
    return arr.reshape(new_shape)


def interval_key(interval):
    return (interval.chrom, interval.start, interval.stop)


def interval_score_pairs(intervals, scores, merge_type):
    return (izip(intervals, scores) if merge_type is None
            else merged_scores(scores, intervals, merge_type))


def _write_2D_deeplift_track(scores, intervals, file_prefix, reorder,
                             categories, merge_type=None):
    # Writes out track as a quantitative category series:
    # http://wiki.wubrowse.org/QuantitativeCategorySeries
    # TODO: implement reorder = False
    if not reorder:
        raise NotImplementedError

    assert scores.ndim == 3

    logger.info('Writing 2D track of shape: {}'.format(scores.shape))
    logger.info('Writing to file: {}'.format(file_prefix))

    if categories is None:
        categories = np.arange(scores.shape[1])

    with open(file_prefix, 'w') as fp:
        line_id = 0
        for interval, score in interval_score_pairs(intervals, scores,
                                                    merge_type):
            line_id = write_score_for_single_interval(fp, interval, score,
                                                      line_id, categories)

    logger.info('Wrote hammock file.')

    compressed_file = tabix_index(file_prefix, preset='bed', force=True)
    assert compressed_file == file_prefix + '.gz'
    logger.info('Compressed and indexed hammock file.')


def merged_scores(scores, intervals, merge_type):
    # A generator that returns merged intervals/scores
    # Scores should have shape: #examples x #categories x #interval_size
    # Second dimension can be omitted for a 1D signal
    signal_dims = scores.ndim - 1
    assert signal_dims in {1, 2}

    # Only support max for now
    assert merge_type == 'max'
    score_first_dim = 1 if signal_dims == 1 else scores.shape[1]

    dtype = scores.dtype

    sort_idx, sorted_intervals = \
        zip(*sorted(enumerate(intervals),
                    key=lambda item: interval_key(item[1])))
    sorted_intervals = BedTool(sorted_intervals)

    # Require at least 1bp overlap
    # Explicitly convert to list otherwise it will keep opening a file when
    # retrieving an index resulting in an error (too many open files)
    interval_clust = list(sorted_intervals.cluster(d=-1))
    for _, group in groupby(izip(sort_idx, interval_clust),
                            key=lambda item: item[1].fields[-1]):
        idx_interval_pairs = list(group)
        group_idx, group_intervals = zip(*idx_interval_pairs)

        if len(idx_interval_pairs) == 1:
            yield group_intervals[0], scores[group_idx[0], ...]
        else:
            group_chrom = group_intervals[0].chrom
            group_start = min(interval.start for interval in group_intervals)
            group_stop = max(interval.stop for interval in group_intervals)

            # This part needs to change to support more merge_types (e.g. mean)
            group_score = np.full((score_first_dim, group_stop - group_start),
                                  -np.inf, dtype)
            for idx, interval in idx_interval_pairs:
                slice_start = interval.start - group_start
                slice_stop = slice_start + (interval.stop - interval.start)
                group_score[..., slice_start:slice_stop] = \
                    np.maximum(group_score[..., slice_start:slice_stop],
                               scores[idx, ...])
            if signal_dims == 1:
                group_score = group_score.squeeze(axis=0)
            yield Interval(group_chrom, group_start, group_stop), group_score


def _write_1D_deeplift_track(scores, intervals, file_prefix, merge_type=None, 
                             CHROM_SIZES='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes'):
    assert scores.ndim == 2

    bedgraph = file_prefix + '.bedGraph'
    bigwig = file_prefix + '.bw'

    logger.info('Writing 1D track of shape: {}'.format(scores.shape))
    logger.info('Writing to file: {}'.format(bigwig))

    with open(bedgraph, 'w') as fp:
        for interval, score in interval_score_pairs(intervals, scores,
                                                    merge_type):
            chrom = interval.chrom
            start = interval.start
            for score_idx, val in enumerate(score):
                fp.write('%s\t%d\t%d\t%g\n' % (chrom,
                                               start + score_idx,
                                               start + score_idx + 1,
                                               val))
    logger.info('Wrote bedgraph.')

    try:
        output = subprocess.check_output(
            ['wigToBigWig', bedgraph, CHROM_SIZES, bigwig],
            stderr=subprocess.STDOUT)
        logger.info('wigToBigWig output: {}'.format(output))
    except subprocess.CalledProcessError as e:
        logger.error('wigToBigWig terminated with exit code {}'.format(
            e.returncode))
        logger.error('output was:\n' + e.output)

    logger.info('Wrote bigwig.')


def write_deeplift_track(scores, intervals, file_prefix, reorder=True,
                         categories=None, merge_type=None,
                         CHROM_SIZES='/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes'):
    if len(scores.shape) != BLOB_DIMS:
        raise ValueError('scores should have same number of dims as a blob')
    if scores.shape[0] != len(intervals):
        raise ValueError('intervals list should have the same number of '
                         'elements as the number of rows in scores')
    if merge_type not in {'max', None}:
        raise ValueError('invalid merge type {}'.format(merge_type))

    # don't squeeze out the first (samples) dimension
    squeezable_dims = tuple(dim for dim, size in enumerate(scores.shape)
                            if size == 1 and dim > 0)
    scores = scores.squeeze(axis=squeezable_dims)
    signal_dims = scores.ndim - 1
    if signal_dims == 2:
        _write_2D_deeplift_track(scores, intervals, file_prefix,
                                 categories=categories, reorder=reorder,
                                 merge_type=merge_type)
    elif signal_dims == 1:
        _write_1D_deeplift_track(scores, intervals, file_prefix,
                                 merge_type=merge_type,
                                 CHROM_SIZES=CHROM_SIZES)
    else:
        raise ValueError('Cannot handle scores with {} signal dims;'
                         'Only 1D/2D signals supported'.format(signal_dims))


def partition_intervals(intervals):
    # Partition interval list into several lists of non-overlapping intervals
    # We need this because of subpeaks whose windows will overlap
    # partitions will contain a list of partitions, which is itself a list
    # containing (index, interval) pairs, where the index is such that
    # intervals[index] = interval for the original intervals list passed in.
    from pybedtools import Interval

    def overlap(i1, i2):
        if i1.chrom != i2.chrom:
            return False
        return (i2.start < i1.end and i2.end > i1.start)

    partitions = []
    # sort intervals to get a list of (index, interval) pairs such that
    # intervals[index] = interval
    remaining = sorted(
        enumerate(intervals),
        key=lambda item: (item[1].chrom, item[1].start, item[1].stop))

    while remaining:
        nonoverlapping = [(-1, Interval('sentinel', 0, 0))]
        overlapping = []

        for idx, interval in remaining:
            if not overlap(nonoverlapping[-1][1], interval):
                nonoverlapping.append((idx, interval))
            else:
                overlapping.append((idx, interval))

        partitions.append(nonoverlapping[1:])
        remaining = overlapping

    return partitions


def plot_tracks_merged(scores, intervals, output_prefix, merge_type='max'):
    # supports bedGraph and quantitative category tracks
    # Note: intervals must be 1bp in length (i.e. there is an entry per-base)
    # write out entire track
    # sort intervals, then coalesce
    logger.info('Writing tracks...')

    # Make the individual tracks
    for score_type, scoredict in scores.iteritems():
        for track_name, track_scores in scoredict.iteritems():
            logger.info('Writing track: {} {}'.format(track_name,
                                                      score_type))
            file_prefix = '{}_{}_{}_{}'.format(output_prefix, track_name,
                                               score_type, merge_type)
            write_deeplift_track(track_scores, intervals, file_prefix,
                                 merge_type=merge_type)


def plot_tracks(scores, intervals, output_prefix):
    # scores should be a dict mapping score_type to actual scores
    interval_partitions = partition_intervals(intervals)
    logger.info('Intervals partitioned into {} non-overlapping partitions'
                .format(len(interval_partitions)))

    logger.info('Writing tracks...')
    for partition_idx, partition in enumerate(interval_partitions):
        logger.info('On partition {} ({} intervals)'.format(partition_idx,
                                                            len(partition)))
        indices, intervals = zip(*partition)

        # Make the individual tracks
        for score_type, scoredict in scores.iteritems():
            for track_name, track_scores in scoredict.iteritems():
                logger.info('Writing track: {} {}'.format(track_name,
                                                          score_type))
                file_prefix = '{}_{}_{}_{}'.format(output_prefix, track_name,
                                                   score_type, partition_idx)
                write_deeplift_track(track_scores[indices, ...], intervals,
                                     file_prefix)
