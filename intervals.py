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


def pad_interval(interval, interval_size):
    """
    Interval padding utility for BedTool.each

    Returns padded interval if at least half of target size.
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
    # group last region windows
    window_numbers = np.array([window.count for window in windows])
    first_window_indxs = np.where(window_numbers == 1)[0]
    last_window_indxs = first_window_indxs[1:] - 1
    last_windows = windows.at(last_window_indxs)
    # group all other windows
    non_last_window_indxs = list(set(range(len(window_numbers)))
                                 - set(last_window_indxs))
    non_last_windows = windows.at(non_last_window_indxs)
    # pad last windows
    padded_last_windows = last_windows.each(pad_interval, bin_size)
    # combine both sets of windows
    return combine_bedtools([non_last_windows, padded_last_windows])
