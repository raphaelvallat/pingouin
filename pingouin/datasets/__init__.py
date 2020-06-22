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
        Requested dataset.

    Examples
    --------
    Load the `Penguin <https://github.com/allisonhorst/palmerpenguins>`_
    dataset:

    >>> import pingouin as pg
    >>> df = pg.read_dataset('penguins')
    >>> df # doctest: +SKIP
        species  island  bill_length_mm  ...  flipper_length_mm  body_mass_g     sex
    0    Adelie  Biscoe            37.8  ...              174.0       3400.0  female
    1    Adelie  Biscoe            37.7  ...              180.0       3600.0    male
    2    Adelie  Biscoe            35.9  ...              189.0       3800.0  female
    3    Adelie  Biscoe            38.2  ...              185.0       3950.0    male
    4    Adelie  Biscoe            38.8  ...              180.0       3800.0    male
    ..      ...     ...             ...  ...                ...          ...     ...
    339  Gentoo  Biscoe             NaN  ...                NaN          NaN     NaN
    340  Gentoo  Biscoe            46.8  ...              215.0       4850.0  female
    341  Gentoo  Biscoe            50.4  ...              222.0       5750.0    male
    342  Gentoo  Biscoe            45.2  ...              212.0       5200.0  female
    343  Gentoo  Biscoe            49.9  ...              213.0       5400.0    male
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

    Returns
    -------
    datasets : :py:class:`pandas.DataFrame`
        A dataframe with the name, description and reference of all the
        datasets included in Pingouin.

    Examples
    --------

    >>> import pingouin as pg
    >>> all_datasets = pg.list_dataset()
    >>> all_datasets.index.tolist()
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
     'rm_missing',
     'tips']
     """
    return dts.set_index('dataset')
