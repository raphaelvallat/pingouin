# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
import pandas as pd
from itertools import combinations
from pingouin import (compute_effsize, _remove_rm_na, _extract_effects,
                      multicomp, _export_table)

__all__ = ["pairwise_ttests", "pairwise_corr"]


def _append_stats_dataframe(stats, x, y, xlabel, ylabel, effects, alpha,
                            paired, df_ttest, ef, eftype, time=np.nan):
    stats = stats.append({
        'A': xlabel,
        'B': ylabel,
        'mean(A)': np.round(np.mean(x), 3),
        'mean(B)': np.round(np.mean(y), 3),
        # Use ddof=1 for unibiased estimator (pandas default)
        'std(A)': np.round(np.std(x, ddof=1), 3),
        'std(B)': np.round(np.std(y, ddof=1), 3),
        'Type': effects,
        'Paired': paired,
        'tail': df_ttest.loc['T-test', 'tail'],
        # 'Alpha': alpha,
        'T-val': df_ttest.loc['T-test', 'T-val'],
        'p-unc': df_ttest.loc['T-test', 'p-val'],
        'BF10': df_ttest.loc['T-test', 'BF10'],
        'efsize': ef,
        'eftype': eftype,
        'Time': time}, ignore_index=True)
    return stats


def pairwise_ttests(dv=None, between=None, within=None, effects='all',
                    data=None, alpha=.05, tail='two-sided', padjust='none',
                    effsize='hedges', return_desc=False, export_filename=None):
    '''Pairwise T-tests.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between: string
        Name of column containing the between factor.
    within : string
        Name of column containing the within factor.
    data : pandas DataFrame
        DataFrame
    alpha : float
        Significance level
    tail : string
        Indicates whether to return the 'two-sided' or 'one-sided' p-values
    padjust : string
        Method used for testing and adjustment of pvalues.
        Available methods are ::

        'none' : no correction
        'bonferroni' : one-step Bonferroni correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    effsize : string or None
        Effect size type. Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
    return_desc : boolean
        If True, return group means and std
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'A' : Name of first measurement
        'B' : Name of second measurement
        'Paired' : indicates whether the two measurements are paired or not
        'Tail' : indicate whether the p-values are one-sided or two-sided
        'T-val' : T-values
        'p-unc' : Uncorrected p-values
        'p-corr' : Corrected p-values
        'p-adjust' : p-values correction method
        'BF10' : Bayes Factor
        'efsize' : effect sizes
        'eftype' : type of effect size

    Examples
    --------
    Compute Bonferroni-corrected pairwise post-hocs T-tests from a mixed model
    design.

        >>> import pandas as pd
        >>> from pingouin import pairwise_ttests, print_table
        >>> df = pd.read_csv('dataset.csv')
        >>> post_hocs = pairwise_ttests(dv='DV', within='Time',
        >>>                        between='Group', data=df, effects='all',
        >>>                        padjust='bonf', effsize='hedges')
        >>> # Print the table with 3 decimals
        >>> print_table(post_hocs, floatfmt=".3f")
    '''
    from pingouin.parametric import ttest
    effects = 'within' if between is None else effects
    effects = 'between' if within is None else effects

    if tail not in ['one-sided', 'two-sided']:
        raise ValueError('Tail not recognized')

    if not isinstance(alpha, float):
        raise ValueError('Alpha must be float')

    # Remove NAN in repeated measurements
    if within is not None and data[dv].isnull().values.any():
        data = _remove_rm_na(dv=dv, within=within, data=data)

    # Extract main effects
    dt_array, nobs = _extract_effects(dv=dv, between=between, within=within,
                                      effects=effects, data=data)

    stats = pd.DataFrame([])

    # OPTION A: simple main effects
    if effects.lower() in ['within', 'between']:
        # Compute T-tests
        paired = True if effects == 'within' else False

        # Extract column names
        col_names = list(dt_array.columns.values)

        # Number and labels of possible comparisons
        if len(col_names) >= 2:
            combs = list(combinations(col_names, 2))
            # ntests = len(combs)
        else:
            raise ValueError('Data must have at least two columns')

        # Initialize vectors
        for comb in combs:
            col1, col2 = comb
            x = dt_array[col1].dropna().values
            y = dt_array[col2].dropna().values
            df_ttest = ttest(x, y, paired=paired, tail=tail)
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
            stats = _append_stats_dataframe(stats, x, y, col1, col2, effects,
                                            alpha, paired, df_ttest, ef,
                                            effsize)

    # OPTION B: interaction
    if effects.lower() == 'interaction':
        paired = False
        for time, sub_dt in dt_array.groupby(level=0, axis=1):
            col1, col2 = sub_dt.columns.get_level_values(1)
            x = sub_dt[(time, col1)].dropna().values
            y = sub_dt[(time, col2)].dropna().values
            df_ttest = ttest(x, y, paired=paired, tail=tail)
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
            stats = _append_stats_dataframe(stats, x, y, col1, col2, effects,
                                            alpha, paired, df_ttest, ef,
                                            effsize, time)

    if effects.lower() == 'all':
        stats_within = pairwise_ttests(dv=dv, within=within, effects='within',
                                       data=data, alpha=alpha, tail=tail,
                                       padjust=padjust, effsize=effsize,
                                       return_desc=return_desc)
        stats_between = pairwise_ttests(dv=dv, between=between,
                                        effects='between', data=data,
                                        alpha=alpha, tail=tail,
                                        padjust=padjust, effsize=effsize,
                                        return_desc=return_desc)

        stats_interaction = pairwise_ttests(dv=dv, within=within,
                                            between=between,
                                            effects='interaction',
                                            data=data, alpha=alpha, tail=tail,
                                            padjust=padjust, effsize=effsize,
                                            return_desc=return_desc)
        stats = pd.concat([stats_within, stats_between,
                           stats_interaction], sort=False).reset_index()

    # Multiple comparisons
    padjust = None if stats['p-unc'].size <= 1 else padjust
    if padjust is not None:
        if padjust.lower() != 'none':
            reject, stats['p-corr'] = multicomp(stats['p-unc'].values,
                                                alpha=alpha, method=padjust)
            stats['p-adjust'] = padjust
            # stats['reject'] = reject
    else:
        stats['p-corr'] = None
        stats['p-adjust'] = None
        # stats['reject'] = stats['p-unc'] < alpha

    # stats['reject'] = stats['reject'].astype(bool)
    stats['Paired'] = stats['Paired'].astype(bool)

    # Reorganize column order
    col_order = ['Type', 'Time', 'A', 'B', 'mean(A)', 'std(A)', 'mean(B)',
                 'std(B)', 'Paired', 'T-val', 'tail', 'p-unc',
                 'p-corr', 'p-adjust', 'BF10', 'efsize', 'eftype']

    if not return_desc and effects.lower() != 'all':
        stats.drop(columns=['mean(A)', 'mean(B)', 'std(A)', 'std(B)'],
                   inplace=True)

    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats


def pairwise_corr(data, columns=None, tail='two-sided', method='pearson',
                  padjust='none', export_filename=None):
    '''Pairwise correlations between columns of a pandas dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    columns : list
        Names of columns in data containing the all the dependant variables.
        If columns is None, compute the pairwise correlations on all columns.
    tail : string
        Indicates whether to return the 'two-sided' or 'one-sided' p-values
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::

        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
    padjust : string
        Method used for testing and adjustment of pvalues.
        Available methods are ::

        'none' : no correction
        'bonferroni' : one-step Bonferroni correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'X' : Name(s) of first columns
        'Y' : Name(s) of second columns
        'method' : method used to compute the correlation
        'tail' : indicates whether the p-values are one-sided or two-sided
        'r' : Correlation coefficients
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared values
        'adj_r2' : Adjusted R-squared values
        'z' : Standardized correlation coefficients
        'p-unc' : uncorrected one or two tailed p-values
        'p-corr' : corrected one or two tailed p-values
        'p-adjust' : Correction method

    Notes
    -----
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Correlations of -1 or +1 imply
    an exact linear relationship.

    The Spearman correlation is a nonparametric measure of the monotonicity of
    the relationship between two datasets. Unlike the Pearson correlation,
    the Spearman correlation does not assume that both datasets are normally
    distributed. Correlations of -1 or +1 imply an exact monotonic
    relationship.

    Kendall’s tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate
    strong disagreement.

    The percentage bend correlation (Wilcox 1994) is a robust method that
    protects against univariate outliers.

    The Shepherd's pi correlation (Schwarzkopf et al. 2012) is a robust method
    that returns the equivalent of the Spearman's rho after outliers removal.

    Please note that NaN are automatically removed from datasets.

    Examples
    --------
    1. One-tailed spearman correlation corrected for multiple comparisons

        >>> import pandas as pd
        >>> from pingouin import pairwise_corr, print_table
        >>> data = pd.read_csv('mydata.csv')
        >>> stats = pairwise_corr(data, method='spearman', tail='two-sided',
        >>>                       padjust='bonf')
        >>> print_table(stats)

    2. Robust two-sided correlation with uncorrected p-values

        >>> import pandas as pd
        >>> from pingouin import pairwise_corr, print_table
        >>> data = pd.read_csv('mydata.csv')
        >>> stats = pairwise_corr(data, columns=['Col1', 'Col2', 'Col3'],
        >>>         method='percbend', tail='two-sided')
        >>> print_table(stats)

    3. Export the results to a .csv file

        >>> import pandas as pd
        >>> from pingouin import pairwise_corr, print_table
        >>> data = pd.read_csv('mydata.csv')
        >>> pairwise_corr(data, columns=['Col1', 'Col2', 'Col3'],
        >>>         method='kendall', export_filename='pairwise_corr.csv')
    '''
    from pingouin.correlation import corr
    from pingouin.effsize import compute_esci

    if tail not in ['one-sided', 'two-sided']:
        raise ValueError('Tail not recognized')

    # Initialize empty DataFrame
    stats = pd.DataFrame()

    # Combinations between columns
    if columns is None:
        columns = data.keys().values
    combs = list(combinations(columns, 2))

    # Initialize vectors
    for comb in combs:
        col1, col2 = comb
        x = data[col1].values
        y = data[col2].values
        cor_st = corr(x, y, tail=tail, method=method).reset_index(drop=True)
        stats = stats.append({
            'X': col1,
            'Y': col2,
            'method': method,
            'tail': tail,
            'r': cor_st['r'][0],
            'CI95%': compute_esci(ef=cor_st['r'][0], nx=len(x), ny=len(y)),
            'r2': cor_st['r2'][0],
            'adj_r2': cor_st['adj_r2'][0],
            'p-unc': cor_st['p-val'][0],
            'BF10': cor_st['BF10'][0] if method == 'pearson' else np.nan},
            ignore_index=True)

    # Multiple comparisons
    padjust = None if stats['p-unc'].size <= 1 else padjust
    if padjust is not None:
        if padjust.lower() != 'none':
            reject, stats['p-corr'] = multicomp(stats['p-unc'].values,
                                                method=padjust)
            stats['p-adjust'] = padjust
    else:
        stats['p-corr'] = None
        stats['p-adjust'] = None

    # Standardize correlation coefficients (Fisher z-transformation)
    stats['z'] = np.arctanh(stats['r'].values)

    # Round values
    for c in ['r', 'r2', 'adj_r2', 'z']:
        stats[c] = stats[c].round(3)

    col_order = ['X', 'Y', 'method', 'tail', 'r', 'CI95%', 'r2', 'adj_r2',
                 'z', 'p-unc', 'p-corr', 'p-adjust', 'BF10']
    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats
