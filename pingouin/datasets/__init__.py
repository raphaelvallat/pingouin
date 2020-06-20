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
    data : :py:class:`pandas.DataFrame`
        Dataset

    Examples
    --------
    Load the ANOVA dataset

    >>> import pingouin as pg
    >>> df = read_dataset('penguins')
    """
    # Check extension
    d, ext = op.splitext(dname)
    if ext.lower() == '.csv':
        dname = d
    # Check that dataset exist
    if dname not in dts['dataset'].to_numpy():
        raise ValueError('Dataset does not exist. Valid datasets names are',
                         dts['dataset'].to_numpy())
    # Load dataset
    return pd.read_csv(op.join(ddir, dname + '.csv'), sep=',')


def list_dataset():
    """List available example datasets.

    Examples
    --------

    >>> import pingouin as pg
    >>> all_datasets = pg.list_dataset()
    >>> all_datasets.index
    ['ancova',
     'anova',
     'anova2',
     'anova2_unbalanced',
     'anova3',
     'anova3_unbalanced',
     'chi2_independence',
     'chi2_mcnemar',
     'circular',
     'cochran',
     'cronbach_alpha',
     'cronbach_wide_missing',
     'icc',
     'mediation',
     'mixed_anova',
     'mixed_anova_unbalanced',
     'multivariate',
     'pairwise_corr',
     'pairwise_ttests',
     'pairwise_ttests_missing',
     'partial_corr',
     'penguins',
     'rm_anova',
     'rm_anova_wide',
     'rm_anova2',
     'rm_corr',
     'rm_missing']
     """
    return dts.set_index('dataset')
