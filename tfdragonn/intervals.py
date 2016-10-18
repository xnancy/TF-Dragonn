import numpy as np
import pybedtools
from pybedtools import BedTool
from joblib import Parallel, delayed
from builtins import zip

def bed_intersection_labels(region_bedtool, feature_bedtool, f=0.5, F=0.5, e=True, **kwargs):
    """
    intersects regions with feature bed and returns binary labls
    """
    region_bedtool = BedTool(region_bedtool)
    if feature_bedtool is not None:
        try:
            overlap_counts = [interval.count for interval in
                              region_bedtool.intersect(BedTool(feature_bedtool), c=True, f=f, F=F, e=e, **kwargs)]
        except: # handle unexpected field numbers in feature bedtool by truncating it to bed3
            feature_df = BedTool(feature_bedtool).to_dataframe()
            feature_bedtool = BedTool.from_dataframe(feature_df.iloc[:,[0,1,2]])
            overlap_counts = [interval.count for interval in
                              region_bedtool.intersect(feature_bedtool, c=True, f=f, F=F, e=e, **kwargs)]
        labels = np.array(overlap_counts) > 0
        return labels.astype(int)[:, np.newaxis]
    else:
        return -1 * np.ones((region_bedtool.count(), 1))


def multibed_intersection_labels(region_bedtool, feature_bedtools, f=0.5, F=0.5, e=True, **kwargs):
    """
    intersects regions with all feature beds and returns binary labels
    """
    labels = [bed_intersection_labels(region_bedtool, feature_bedtool, f=f, F=F, e=e, **kwargs)
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


def get_tf_predictive_setup(true_feature_bedtools, region_bedtool=None,
                            bin_size=200, flank_size=400, stride=50,
                            filter_flank_overlaps=True, n_jobs=1,
                            ambiguous_feature_bedtools=None,
                            genome='hg19', save_to_prefix=None):
    """
    Implements the tf (and general) imputation data setup for a single sample.
    TODOs
        support chrom.sizes file for personal genomes

    Parameters
    ----------
    tf_feature_peak_bedtools : list of filenames, BedTools or None items
        None items are treated as missing data.
    region_bedtools : filename or BedTool, optional
        If not set, union of tf_feature_peak_bedtools is used.
    filter_flank_overlaps : bool, default: True
        Labels negative bins whose flanks overlap target regions as ambiguous.
    ambiguous_feature_bedtools : list of filenames, BedTools or None items, optional
    genome : str, default: 'hg19'
        Can be any genome name supported by pybedtools.
    """
    # initialize feature bedtools
    true_feature_bedtools = [BedTool(bedtool) for bedtool in true_feature_bedtools
                             if bedtool is not None]
    # sanity checks
    if ambiguous_feature_bedtools is not None:
        assert len(ambiguous_feature_bedtools) == len(true_feature_bedtools)
        ambiguous_feature_bedtools = [BedTool(ambiguous_feature_bedtool)
                                      for ambiguous_feature_bedtool in ambiguous_feature_bedtools]
    # bin region_bedtools
    if region_bedtool is not None:
        bins = bin_bed(BedTool(region_bedtool), bin_size=bin_size, stride=stride)
    else: # use union of true peak bedtools
        bedtools_to_merge = [bedtool for bedtool in true_feature_bedtools if bedtool is not None]
        region_bedtool = BedTool.cat(*bedtools_to_merge, postmerge=True, force_truncate=True)
        bins = bin_bed(region_bedtool, bin_size=bin_size, stride=stride)
    # filter bins to chr1-22,X,Y
    chrom_list = ["chr%i" % (i) for i in range(1, 23)]
    chrom_list += ["chrX", "chrY"]
    bins = BedTool(bins).each(filter_interval_by_chrom, chrom_list)
    bins = bins.saveas() # save to temp file to enable counting
    num_bins = bins.count()
    # set genome to hg19
    bins = bins.set_chromsizes(genome)
    # intersect bins and tf_true_peaks for true labels
    if n_jobs == 1:
        true_labels_list = []
        for true_feature_bedtool in true_feature_bedtools:
            true_labels = bed_intersection_labels(bins, true_feature_bedtool)
            true_labels_list.append(true_labels)
    elif n_jobs > 1: # multiprocess bed intersections
        # save feature bedtools in temp files. Note: not necessary when inputs are filnames
        true_feature_bedtools = [bedtool.saveas() for bedtool in true_feature_bedtools]
        true_labels_list = Parallel(n_jobs=n_jobs)(delayed(bed_intersection_labels)(bins.fn, bedtool.fn)
                                                   for bedtool in true_feature_bedtools)
    true_labels = np.concatenate(true_labels_list, axis=1)
    bins_and_flanks = bins.slop(b=flank_size)
    if filter_flank_overlaps:
        # intersect bins and flanks for any overlap  with true features
        if n_jobs == 1:
            flank_labels_list = []
            for true_feature_bedtool in true_feature_bedtools:
                flank_labels = bed_intersection_labels(bins, true_feature_bedtool, f=10**-9, F=10**-9)
                flank_labels_list.append(flank_labels)
        elif n_jobs > 1:
            flank_labels_list = Parallel(n_jobs=n_jobs)(delayed(bed_intersection_labels)(bins.fn, bedtool.fn, f=10**-9, F=10**-9)
                                                        for bedtool in true_feature_bedtools)
        flank_labels = np.concatenate(flank_labels_list, axis=1)
        # we label negative bins with any flank overlap with true features as ambiguous
        true_labels[(true_labels == 0) * (flank_labels == 1)] = -1
    if ambiguous_feature_bedtools is not None:
        # intersect bins and ambiguous tfs for ambiguous labels
        if n_jobs == 1:
            ambiguous_labels_list = []
            for ambiguous_feature_bedtool in ambiguous_feature_bedtools:
                ambiguous_labels = bed_intersection_labels(bins, ambiguous_feature_bedtool)
                ambiguous_labels_list.append(ambiguous_labels)
        elif n_jobs > 1:
            ambiguous_feature_bedtools = [bedtool.saveas() for bedtool in ambiguous_feature_bedtools]
            ambiguous_labels_list = Parallel(n_jobs=n_jobs)(delayed(bed_intersection_labels)(bins.fn, bedtool.fn)
                                                            for bedtool in ambiguous_feature_bedtools)
        ambiguous_labels = np.concatenate(ambiguous_labels_list, axis=1)
        # we label negative bins that overlap ambiguous feature as ambiguous
        true_labels[(true_labels == 0) * (ambiguous_labels == 1)] = -1
        # TODO: do we want to also filter based on any flank overlap with ambiguous features??
    if save_to_prefix is not None: # save intervals and labels
        intervals_fname = "%s.intervals.bed" % (save_to_prefix)
        labels_fname = "%s.labels.npy" % (save_to_prefix)
        bins_and_flanks = bins_and_flanks.saveas().moveto(intervals_fname)
        np.save(labels_fname, true_labels)
    return bins_and_flanks, true_labels


def train_test_chr_split(intervals, labels, test_chr):
    train_intervals = []
    train_labels = []
    test_intervals = []
    test_labels = []
    for interval, label in zip(intervals, labels):
        if interval.chrom in test_chr:
            test_intervals.append(interval)
            test_labels.append(label)
        else:
            train_intervals.append(interval)
            train_labels.append(label)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_intervals, test_intervals, train_labels, test_labels
