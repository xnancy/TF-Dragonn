from __future__ import absolute_import, division, print_function

import json
import math
import numpy as np
import ntpath
import os

from genomedatalayer.util import makedirs
from pybedtools import BedTool, Interval

def extract_expression_to_file(datafile, output_dir, components_file, replicate_datafile=None,
                               dtype=np.float32, overwrite=False):
    """
    Processes mean expression TPMs with asinh
    followed by z-scores projected to a normal distribution,
    and then projects to principal components.

    Parameters
    ----------
    datafile: tsv file
    components_file: .npy file
    replicate_datafile: tsv file for a replicate, optional
    """
    makedirs(output_dir, exist_ok=overwrite)
    components = np.load(components_file)
    data = np.loadtxt(datafile, dtype=str)
    tpm_col_indx = np.where(data[0, :] == 'TPM')[0]
    tpm = data[1:, tpm_col_indx].astype(float)
    if replicate_datafile is not None: # get mean tpms
        data2 = np.loadtxt(replicate_datafile, dtype=str)
        tpm2 = data2[1:, tpm_col_indx].astype(float)
        tpm = (tpm + tpm2) / 2
    tpm_asinh = np.arcsinh(tpm)
    mean = tpm_asinh.mean(axis=0)
    std = tpm_asinh.std(axis=0)
    tpm_asinh_zscored = (tpm_asinh - mean) / std
    tpm_asinh_zscored_normal = np.exp(-0.5*(tpm_asinh_zscored)**2) / math.sqrt(2*math.pi)
    
    projected_tpm = np.dot(tpm_asinh_zscored_normal.T, components.T)[0, :]
    np.save(os.path.join(output_dir, "features.npy"), projected_tpm.astype(dtype))

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
        json.dump({'file_shapes': projected_tpm.shape,
                   'type': 'constant_numpy_array',
                   'components_file': components_file,
                   'replicate_datafile': replicate_datafile,
                   'source': datafile}, fp)


def write_tss_expression_bedgraph(tss_gff, expression_tsv, output_dir,
                                  expression_tsv2=None, overwrite=False):
    """
    Writes bedGraph file with tss intervals and their expression TPMs in the 4th column.

    Parameters
    ----------
    tss_gff : tss GFF file
    expression_tsv : tsv file
        E.g. output from RSEM pipeline.
    expression_tsv2 : tsv file, optional
        Writes mean expression tpms if this file is provided.
    """
    makedirs(output_dir, exist_ok=overwrite)
    data = np.loadtxt(expression_tsv, dtype=str)
    tpm_col_indx = np.where(data[0, :] == 'TPM')[0]
    tpm = data[1:, tpm_col_indx].astype(float)
    if expression_tsv2 is not None: # get mean tpms
        data2 = np.loadtxt(expression_tsv2, dtype=str)
        tpm2 = data2[1:, tpm_col_indx].astype(float)
        tpm = (tpm + tpm2) / 2

    tss_bedtool = BedTool(tss_gff)
    # check that gene ids match
    gene_id_col_indx = np.where(data[0, :] == 'gene_id')[0]
    gene_ids = data[1:, gene_id_col_indx]
    bedtool_df = tss_bedtool.to_dataframe()
    bedtool_gene_ids = [k.split()[1] for k in bedtool_df.iloc[:, -1]]
    bedtool_gene_ids = np.asarray(bedtool_gene_ids)
    assert np.all(np.in1d(bedtool_gene_ids, gene_ids))

    gene_id2tpm = {gene_id: float(tpm) for gene_id, tpm in np.hstack((gene_ids, tpm))}
    intervals = [Interval(interval.chrom, interval.start, interval.stop, str(gene_id2tpm[gene_id]))
                 for interval, gene_id in zip(tss_bedtool, bedtool_gene_ids)]
    BedTool(intervals).saveas(os.path.join(output_dir, "data.bedGraph"))

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
        json.dump({'type': 'intervals_data_bed',
                   'tss_gff': tss_gff,
                   'expression_tsv': expression_tsv,
                   'expression_tsv2': expression_tsv2}, fp)
