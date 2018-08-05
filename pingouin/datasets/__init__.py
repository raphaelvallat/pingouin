import pandas as pd
import os.path as op
from pingouin.utils import print_table

dts = pd.DataFrame({'dataset': ['bland1995', 'mcclave1991'],
                    '# rows': [47, 19],
                    '# cols': [3, 3],
                    'useful for': ['rm_corr', ['anova', 'pairwise_tukey']],
                    'ref': ['Bland & Altman (1995)',
                            'McClave and Dietrich (1991)'],
                     })

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
    Load the bland1995 dataset

        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('bland1995')
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
    ddir = op.dirname(op.realpath(__file__))
    return pd.read_csv(op.join(ddir, dname + '.csv'), sep=',')


def list_dataset():
    """List available example datasets.
    """
    print_table(dts)
