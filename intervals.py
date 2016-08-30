import numpy as np
import pybedtools
from pybedtools import BedTool

def bed_intersection_labels(region_bedtool, feature_bedtool, f=0.5, F=0.5, e=True):
    """
    intersects regions with feature bed and returns binary labls
    """
    overlap_counts = [interval.count for interval in
                      region_bedtool.intersect(feature_bedtool, c=True, f=f, F=F, e=e)]
    labels = np.array(overlap_counts) > 0
    
    return labels.astype(int)[:, np.newaxis]


def multibed_intersection_labels(region_bedtool, feature_bedtools, f=0.5, F=0.5, e=True):
    """
    intersects regions with all feature beds and returns binary labels
    """
    labels = [bed_intersection_labels(region_bedtool, feature_bedtool, f=f, F=F, e=e)
              for feature_bedtool in feature_bedtools]

    return np.concatenate(tuple(labels), axis=1)


def combine_bedtools(bedtools):
    """
    Combines sequence of bedtools.
    """
    def generate_all_intervals():
        for bedtool in bedtools:
            for interval in bedtool:
                yield interval

    return BedTool(generate_all_intervals())


def filter_interval_by_chrom(interval, chrom_list):
    """
    Subsets to intervals in chr list.
    To be used with BedTool.each

    Parameters
    ----------
    chrom_list : list of str
    """
    if interval.chrom in chrom_list:
        return interval
    else:
        return False


def pad_interval(interval, interval_size):
    """
    Interval padding utility for BedTool.each

    Returns padded interval if at least half of target size.
    Intervals with correct size are not modified.
    Otherwise, return False
    """
    length = interval.stop - interval.start
    if length < interval_size / 2.0:
        return False
    else:
        interval.stop += interval_size - length
        return interval


def bin_bed(bedtool, bin_size, stride):
    """
    Bins bed regions.
    """
    windows = bedtool.window_maker(bedtool, w=bin_size, s=stride, i="winnum")
    # pad/remove last window of each interval
    return windows.each(pad_interval, bin_size)
