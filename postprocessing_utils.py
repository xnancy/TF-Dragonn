import numpy as np
import os
import pickle

from sklearn.metrics import precision_recall_curve
from dragonn.synthetic import util

from kmedoids import kmedoids
import csi

def get_fdr_thershold(y_true, y_score, fdr_cutoff):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1- precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return thresholds[cutoff_index]

def cluster_grammars(deeplift_scores, max_n_grammars,
                     seqlet_size=7, seqlet_flank=7, n_jobs=1):
    """
    Performs kmedoids+agglomerative clustering.
    TODO: allow user specified filterings parameters such as
        median/percentile initial seqlet filtering
        size of final seqlet subset to cluster on
        cc_thershold
    **Note**: requires jisraeli fork of modisco as of 9/12/2016. 

    Parameters
    ----------
    deeplift_scores: 4darray
        single task scores from deeplift method.
    max_n_grammars : int

    Returns
    -------
    List of Grammar objects.
    """
    segment_identifier = csi.FixedWindowAroundPeaks(
        slidingWindowForMaxSize=seqlet_size,
        flankToExpandAroundPeakSize=seqlet_flank,
        excludePeaksWithinWindow=seqlet_size + seqlet_flank,
        ratioToTopPeakToInclude=0.5,
        maxSegments=10)
    grammars, grammar_indices = csi.getSeqlets(
        rawDeepLIFTContribs=deeplift_scores,
        indicesToGetSeqletsOn=None,
        outputsBeforeActivation=None,
        activation=None,
        thresholdProb=1.0,
        segmentIdentifier=segment_identifier,
        numThreads=n_jobs,
        secondsBetweenUpdates=6,
        includeNeg=False)
    # throw out grammars smaller than max
    max_grammar_length = max([np.shape(grammar.normedCoreDeepLIFTtrack)[1] for grammar in grammars])
    max_grammar_indices = [i for i, grammar in enumerate(grammars)
                           if np.shape(grammar.normedCoreDeepLIFTtrack)[1]==max_grammar_length]
    grammars = [grammars[i] for i in max_grammar_indices]
    grammar_indices = [grammar_indices[i] for i in max_grammar_indices]
    # subset to grammars with max signal above median max signal
    grammars_sorted = sorted(grammars,
                             key= lambda x: x.normedCoreDeepLIFTtrack.max(),
                             reverse=True)
    above_median_grammars = grammars_sorted[:len(grammars_sorted)/2]
    # cluster a subset of those grammars
    subset_indices = np.random.choice(np.arange(len(above_median_grammars)), size=5000)
    grammars_subset = np.take(above_median_grammars, subset_indices)
    grammars_corr_mat = csi.getCorrelationMatrix(
        grammars_subset,
        normaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
        accountForRevComp=True,
        numThreads=n_jobs,
        secondsBetweenUpdates=6,
        xcorBatchSize=100)
    max_cc = grammars_corr_mat.max()
    min_cc = grammars_corr_mat.min()
    corr_mat_diag = grammars_corr_mat.diagonal()
    assert -0.01 <= min_cc and max_cc <= 1.01, "min cc: %f, max cc: %f" % (min_cc, max_cc)
    grammars_corr_mat *= -1
    grammars_corr_mat += 1
    np.fill_diagonal(grammars_corr_mat, 0)
    labels, _, _ = kmedoids(grammars_corr_mat, n_clusters=max_n_grammars,
                            n_init=10)
    merged_grammars = csi.createMergedGrammars(
        labels,
        grammars_subset,
        normaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
        accountForRevComp=True)
    trimmingFunc = csi.TrimArrayColumnsToNumUnderlyingObs(0.3)
    merged_grammars = csi.adjustGrammarsUsingTrimmingCriterion(
        merged_grammars, trimmingFunc=trimmingFunc)

    def agglomerative_clustering(grammars, cc_threshold):
        grammars_list = grammars.values()
        indices_list = [set([i]) for i in range(len(grammars_list))]
        while True:
            grammars_cc = csi.getCorrelationMatrix(
                grammars_list,
                accountForRevComp=True,
                numThreads=1,
                secondsBetweenUpdates=6,
                xcorBatchSize=None,
                smallerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
                largerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm])

            np.fill_diagonal(grammars_cc, 0)
            max_grammars_cc = np.max(grammars_cc)
            if max_grammars_cc < cc_threshold:
                break
            max_cc_idx1, max_cc_idx2 = np.unravel_index(
                np.argmax(grammars_cc), grammars_cc.shape)
            merged_grammar = grammars_list[max_cc_idx1].merge(
                grammars_list[max_cc_idx2],
                subtracksToInclude=['coreDeepLIFTtrack'], # may need to replace with gradient track
                subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndSdev,
                normaliseFunc=util.CROSSC_NORMFUNC.none,
                smallerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
                largerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
                revComp=True)
            removeset = {max_cc_idx1, max_cc_idx2}
            merged_indices = indices_list[max_cc_idx1].union(indices_list[max_cc_idx2])
            indices_list = [indices_set for i, indices_set in enumerate(indices_list)
                            if i not in removeset]
            indices_list.append(merged_indices)
            grammars_list = [grammar for i, grammar in enumerate(grammars_list)
                             if i not in removeset]
            grammars_list.append(merged_grammar)
        return grammars_list, indices_list

    agglomerated_grammars, agglomerated_indices = agglomerative_clustering(
        merged_grammars, cc_threshold=0.9)

    agglomerated_grammars_dict = {i: grammar for i, grammar in enumerate(agglomerated_grammars)}
    trimmingFunc = csi.TrimArrayColumnsToNumUnderlyingObs(0.3)
    trimmed_agglomerated_grammars = csi.adjustGrammarsUsingTrimmingCriterion(
        agglomerated_grammars_dict, trimmingFunc=trimmingFunc)

    return trimmed_agglomerated_grammars.values()


def pickle_grammars(grammars, fname):
    to_pkl = [(x.normedCoreDeepLIFTtrack, x.numUnderlyingObservations.max())
              for x in grammars]
    pickle.dump(to_pkl, open(fname,'w'))


def unpickle_grammars(fname):
    datas = pickle.load(open(fname))
    dl_tracks = [data[0] for data in datas]
    num_observations = [data[1] for data in datas]
    return dl_tracks, num_observations


def write_dl_scores_bw(intervals, dl_scores, prefix_base, chrom_sizes_fname, task_names=None):
    """
    writes deeplift scores from SequenceClassifier to separate bigwigs for each task.

    Parameters
    ----------
    intervals : bedtool intervals
    dl_scores : 5darray
        output from SequenceClassifier deeplift method.
    prefix_base : str
    chrom_sizes_fname : str
        file with chromosome sizes. TODO: infer these using pybedtools.
    task_names : list, optional 
    """
    if task_names is not None:
        assert len(task_names) == dl_scores.shape[0]
    for _i, sequence_dl_scores in enumerate(dl_scores):
        print("writing scores to bedGraph file..")
        sequence_dl_scores_2d = np.sum(sequence_dl_scores.squeeze(), axis=1)
        prefix = "%s.%s" % (prefix_base,
                            task_names[_i] if task_names is not None else str(_i))
        with open("%s.%s" % (prefix, "bedGraph"), "w") as wf:
            for i, interval in enumerate(intervals):
                chrm = interval.chrom
                start = interval.start
                interval_length = interval.stop - start
                starts = np.asarray(start + np.arange(interval_length), dtype='str')
                stops = np.asarray(start + np.arange(interval_length) + 1, dtype='str')
                for j in np.arange(interval_length):
                    wf.write("%s\t%s\t%s\t%s\n" % (
                        chrm, starts[j], stops[j], str(sequence_dl_scores_2d[i][j])))
        print("processing bedGraph file into bigwig file...")
        os.system("(sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < %s.bedGraph | (sort -u -k1,1 -k2,2n -k3,3n) > %s.bedGraph.max" % (prefix, prefix))
        os.system("rm %s.bedGraph" % (prefix))
        os.system("bedGraphToBigWig %s.bedGraph.max %s %s.bw" % (prefix, chrom_sizes_fname, prefix))
        os.system("rm %s.bedGraph.max" % (prefix))
