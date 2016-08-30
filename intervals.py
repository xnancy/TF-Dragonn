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


def get_tf_predictive_setup(region_bedtool, true_feature_bedtools,
                            bin_size=200, flank_size=400, stride=50,
                            filter_flank_overlaps=True,
                            ambiguous_feature_bedtools=None):
    """
    Implements the tf (and general) imputation data setup for a single sample.
    Assumes genome is hg19.
    TODO: parallelize intersection calls -takes most of the runtime for multiple feature beds.

    Parameters
    ----------
    region_bedtools : list of BedTools
    tf_feature_peak_bedtools : list of BedTool or None elements
    ambiguous_feature_bedtools : list of BedTool or None elements, optional
    """
    # sanity checks
    if ambiguous_feature_bedtools is not None:
        assert len(ambiguous_feature_bedtools) == len(true_feature_bedtools)
    # bin region_bedtools
    bins = bin_bed(region_bedtool, bin_size=bin_size, stride=stride)
    # filter bins to chr1-22,X,Y
    chrom_list = ["chr%i" % (i) for i in range(1, 23)]
    chrom_list += ["chrX", "chrY"]
    bins = BedTool(bins).each(filter_interval_by_chrom, chrom_list)
    bins = bins.saveas() # save to temp file to enable counting
    num_bins = bins.count()
    # set genome to hg19
    bins = bins.set_chromsizes('hg19')
    # intersect bins and tf_true_peaks for true labels
    true_labels_list = []
    for true_feature_bedtool in true_feature_bedtools:
        if true_feature_bedtool is not None:
            true_labels = bed_intersection_labels(bins, true_feature_bedtool)
        else:
            true_labels = -1*np.ones((num_bins, 1))
        true_labels_list.append(true_labels)
    true_labels = np.concatenate(true_labels_list, axis=1)
    bins_and_flanks = bins.slop(b=flank_size)
    if filter_flank_overlaps:
        # intersect bins and flanks for any overlap  with true features
        flank_labels_list = []
        for true_feature_bedtool in true_feature_bedtools:
            if true_feature_bedtool is not None:
                flank_labels = bed_intersection_labels(bins, true_feature_bedtool,
                                                       f=10**-9, F=10**-9)
            else:
                flank_labels = np.zeros((num_bins, 1))
            flank_labels_list.append(flank_labels)
        flank_labels = np.concatenate(flank_labels_list, axis=1)
        ## we label negative bins whose flanks have
        ## any overlap with true features as ambiguous
        neg_bin_indxs = true_labels == 0
        neg_bin_pos_flank_indxs = (neg_bin_indxs*(flank_labels == 1)).astype(bool)
        true_labels[neg_bin_pos_flank_indxs] = -1
    if ambiguous_feature_bedtools is not None:
        # intersect bins and ambiguous tfs for ambiguous labels
        ambg_bin_labels_list = []
        for amb_feature_bedtools in ambiguous_feature_bedtools:
            if amb_feature_bedtools is not None:
                amb_bin_labels = bed_intersection_labels(bins, amb_feature_bedtools)
            else:
                amb_bin_labels = np.zeros((num_bins, 1))
            ambg_bin_labels_list.append(ambg_bin_labels_list)
        ambg_bin_labels = np.concatenate(ambg_bin_labels_list, axis=1)
        # we label negative bins that overlap ambiguous feature as ambiguous
        true_labels[true_labels == 0][ambg_bin_labels == 1] = -1
        # TODO: do we want to also filter based on any flank overlap with ambiguous features??
    return bins_and_flanks, true_labels
