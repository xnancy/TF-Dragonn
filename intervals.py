import numpy as np
import pybedtools
from pybedtools import BedTool
from joblib import Parallel, delayed

def bed_intersection_labels(region_bedtool, feature_bedtool, f=0.5, F=0.5, e=True, **kwargs):
    """
    intersects regions with feature bed and returns binary labls
    """
    region_bedtool = BedTool(region_bedtool)
    if feature_bedtool is not None:
        overlap_counts = [interval.count for interval in
                          region_bedtool.intersect(BedTool(feature_bedtool), c=True, f=f, F=F, e=e, **kwargs)]
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


def bed_intersection_labels_star(args):
        return bed_intersection_labels(*args)


def get_tf_predictive_setup(true_feature_bedtools, region_bedtool=None,
                            bin_size=200, flank_size=400, stride=50,
                            filter_flank_overlaps=True, n_jobs=1,
                            ambiguous_feature_bedtools=None,
                            genome='hg19'):
    """
    Implements the tf (and general) imputation data setup for a single sample.
    TODOs
        multiprocess each set of bed intersections
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
    true_feature_bedtools = [BedTool(true_feature_bedtool)
                             for true_feature_bedtool in true_feature_bedtools]
    # sanity checks
    if ambiguous_feature_bedtools is not None:
        assert len(ambiguous_feature_bedtools) == len(true_feature_bedtools)
        ambiguous_feature_bedtools = [BedTool(ambiguous_feature_bedtool)
                                      for ambiguous_feature_bedtool in ambiguous_feature_bedtools]
    # bin region_bedtools
    if region_bedtool is not None:
        bins = bin_bed(region_bedtool, bin_size=bin_size, stride=stride)
    else: # use union of true peak bedtools
        region_bedtool = BedTool.cat(*true_feature_bedtools, postmerge=True, force_truncate=True)
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
        true_feature_bedtools = [true_feature_bedtool.saveas() for true_feature_bedtool in true_feature_bedtools]
        true_labels_list = Parallel(n_jobs=n_jobs)(delayed(bed_intersection_labels)(bins.fn, true_feature_bedtool.fn)
                                                   for true_feature_bedtool in true_feature_bedtools)
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
