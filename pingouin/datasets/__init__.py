import pandas as pd
import os.path as op
from pingouin.utils import print_table

ddir = op.dirname(op.realpath(__file__))
dts = pd.read_csv(op.join(ddir, 'datasets.csv'), sep=',')

__all__ = ["read_dataset", "list_dataset"]

def read_dataset(dname):
    """Read example datasets.

    Parameters
    ----------
    dname : string
        Name of dataset to read (without extension).
        Must be a valid dataset present in pingouin.datasets

    Returns
    -------
    data : pd.DataFrame
        Dataset

    Examples
    --------
    Load the ANOVA dataset

    >>> from pingouin import read_dataset
    >>> df = read_dataset('anova')
    """
    # Check extension
    d, ext = op.splitext(dname)
    if ext.lower() == '.csv':
        dname = d
    # Check that dataset exist
    if dname not in dts['dataset'].values:
        raise ValueError('Dataset does not exist. Valid datasets names are',
                         dts['dataset'].values)
    # Load dataset
    return pd.read_csv(op.join(ddir, dname + '.csv'), sep=',')


def list_dataset():
    """List available example datasets.

    Examples
    --------

    >>> from pingouin import list_dataset
    >>> list_dataset()  # doctest: +SKIP
    """
    print_table(dts)
