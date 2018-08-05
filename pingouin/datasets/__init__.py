import numpy as np
import pandas as pd
import os

datasets = ['bland1995']

def read_dataset(dname):
    """Read example datasets.

    Parameters
    ----------
    dname : string
        Name of dataset to read.

    Returns
    -------
    data : pd.DataFrame
        Dataset
    """
    # Check extension
    d, ext = os.path.splitext(dname)
    if ext.lower() == '.csv':
        dname = d
    # Check that dataset exist
    if dname not in datasets:
        raise ValueError('Dataset does not exist.')
    # Load dataset
    ddir = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(os.path.join(ddir, dname + '.csv'), sep=',')
