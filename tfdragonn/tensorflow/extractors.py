from __future__ import absolute_import, division, print_function

import json
import math
import numpy as np
import ntpath
import os

from genomedatalayer.util import makedirs

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
    
