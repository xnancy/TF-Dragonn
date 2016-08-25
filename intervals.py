import numpy as np
import pybedtools

def bed_intersection_labels(region_bedtool, feature_bedtool, f=0.5, F=0.5, e=True):
    """
    intersects regions with feature bed and returns binary labls
    """
    overlap_counts = [interval.count for interval in
                      region_bedtool.intersect(feature_bedtool, c=True, f=f, F=F, e=e)]
    labels = np.array(overlap_counts, dtype=int) > 0
    
    return labels.astype(int)[:, np.newaxis]

def multibed_intersection_labels(region_bedtool, feature_bedtools, f=0.5, F=0.5, e=True):
    """
    intersects regions with all feature beds and returns binary labels
    """
    labels = [bed_intersection_labels(region_bedtool, feature_bedtool, f=f, F=F, e=e)
              for feature_bedtool in feature_bedtools]

    return np.concatenate(tuple(labels), axis=1)

def bin_bed(bedtool, bin_size, stride):
    """
    Bins bed regions.
    TODO: pad edge bins (currently do not match bin_size)
    """
    return bedtool.window_maker(bedtool, w=bin_size, s=stride)
