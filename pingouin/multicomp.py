# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
# Code borrowed from statsmodel and mne.stats
import numpy as np
import pandas as pd
from pingouin import (_check_data, _check_dataframe, _extract_effects,
                     compute_effsize)

__all__ = ["fdr", "bonf", "holm", "multicomp", "pairwise_ttests"]


def _ecdf(x):
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR).

    Correction for multiple comparison using FDR.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


def bonf(pvals, alpha=0.05):
    """P-value correction with Bonferroni method.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing
    """
    pvals = np.asarray(pvals)
    pvals_corrected = pvals * float(pvals.size)
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    reject = pvals_corrected < alpha
    return reject, pvals_corrected


def holm(pvals, alpha=0.05):
    """P-value correction with Holm method.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing
    """
    pvals = np.asarray(pvals)
    ntests = pvals.size
    notreject = pvals > alpha / np.arange(ntests, 0, -1)
    nr_index = np.nonzero(notreject)[0]
    if nr_index.size == 0:
        # nonreject is empty, all rejected
        notrejectmin = ntests
    else:
        notrejectmin = np.min(nr_index)
    notreject[notrejectmin:] = True
    reject = ~notreject
    pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
    pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    reject = pvals_corrected < alpha
    return reject, pvals_corrected

def multicomp(pvals, alpha=0.05, method='holm'):
    '''P-values correction for multiple tests.

    Parameters
    ----------
    pvals : array_like
        uncorrected p-values
    alpha : float
        Significance level
    method : string
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are ::

        'bonferroni' : one-step correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction


    Returns
    -------
    reject : array, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : array
        p-values corrected for multiple tests
    '''
    if not isinstance(pvals, (list, np.ndarray)):
        err = "pvals must be a list or a np.ndarray"
        raise ValueError(err)

    if method.lower() in ['b', 'bonf', 'bonferroni']:
        reject, pvals_corrected = bonf(pvals, alpha=alpha)
    elif method.lower() in ['h', 'holm']:
        reject, pvals_corrected = holm(pvals, alpha=alpha)
    elif method.lower() in ['fdr_bh']:
        reject, pvals_corrected = fdr(pvals, alpha=alpha, method='indep')
    elif method.lower() in ['fdr_by']:
        reject, pvals_corrected = fdr(pvals, alpha=alpha, method='negcorr')
    elif method.lower() == 'none':
        return None, None
    else:
        raise ValueError('Multiple comparison method not recognized')
    return reject, pvals_corrected

def _append_stats_dataframe(stats, x, y, xlabel, ylabel, effects, paired, alpha,
                            t, p, ef, eftype, time=np.nan):
    stats = stats.append({
                        'A': xlabel,
                        'B': ylabel,
                        'mean(A)': np.mean(x),
                        'mean(B)': np.mean(y),
                        # Use ddof=1 for unibiased estimator (pandas default)
                        'std(A)': np.std(x, ddof=1),
                        'std(B)': np.std(y, ddof=1),
                        'Type': effects,
                        'Paired': paired,
                        'Alpha': alpha,
                        'T-val': t,
                        'p-unc': p,
                        'Eff_size': ef,
                        'Eff_type': eftype,
                        'Time': time}, ignore_index=True)
    return stats


def pairwise_ttests(dv=None, between=None, within=None, effects='all',
                    data=None, alpha=.05, tail='two-sided', padjust='none',
                    effsize='hedges', return_desc=True):
    '''Pairwise T-tests in Pandas.

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
    p-adjust : string
        Method used for testing and adjustment of pvalues. Available methods are ::

        'none' : no correction
        'bonferroni' : one-step correction
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

    Returns
    -------
    stats : DataFrame
        Stats summary
    '''
    from itertools import combinations
    from scipy.stats import ttest_ind, ttest_rel

    effects = 'within' if between is None else effects
    effects = 'between' if within is None else effects

    if tail not in ['one-sided', 'two-sided']:
        raise ValueError('Tail not recognized')

    if not isinstance(alpha, float):
        raise ValueError('Alpha must be float')

    # Check NA in repeated measurements
    # Works well for one-way repeated measures
    # Needs adaptation for mixed model!
    if within is not None and data[dv].isnull().values.any():
        if effects != 'between':
            rm = list(data[within].unique())
            n_rm = len(rm)
            n_obs = int(data.groupby(within)[dv].count().max())
            data['Subj'] = np.tile(np.arange(n_obs), n_rm)
            data = data.pivot(index='Subj', columns=within, values=dv).dropna()
            data = pd.melt(data, value_vars=rm, var_name=within, value_name=dv)

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
            ntests = len(combs)
        else:
            raise ValueError('Data must have at least two columns')

        # Initialize vectors
        for comb in combs:
            col1, col2 = comb
            x = dt_array[col1].dropna().values
            y = dt_array[col2].dropna().values
            t, p = ttest_rel(x, y) if paired else ttest_ind(x, y)
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
            stats = _append_stats_dataframe(stats, x, y, col1, col2, effects,
                                            paired, alpha, t, p, ef, effsize)

    # OPTION B: interaction
    if effects.lower() == 'interaction':
        paired = False
        for time, sub_dt in dt_array.groupby(level=0, axis=1):
            col1, col2 = sub_dt.columns.get_level_values(1)
            x = sub_dt[(time, col1)].dropna().values
            y = sub_dt[(time, col2)].dropna().values
            t, p = ttest_rel(x, y) if paired else ttest_ind(x, y)
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
            stats = _append_stats_dataframe(stats, x, y, col1, col2, effects,
                                            paired, alpha, t, p, ef, effsize,
                                            time)

    if effects.lower() == 'all':
        stats_within = pairwise_ttests(dv=dv, within=within, effects='within',
                            data=data, alpha=alpha, tail=tail, padjust=padjust,
                            effsize=effsize)
        stats_between = pairwise_ttests(dv=dv, between=between,
                            effects='between', data=data, alpha=alpha,
                            tail=tail, padjust=padjust, effsize=effsize)

        stats_interaction = pairwise_ttests(dv=dv, within=within,
                            between=between, effects='interaction', data=data,
                            alpha=alpha, tail=tail, padjust=padjust,
                            effsize=effsize)
        stats = pd.concat([stats_within, stats_between,
                                stats_interaction]).reset_index()

    # Tail and multiple comparisons
    if tail == 'one-sided':
        stats['p-unc'] *= .5
        stats['Tail'] = 'one-sided'
    else:
        stats['Tail'] = 'two-sided'

    # Multiple comparisons
    if padjust is not None or padjust.lower() is not 'none':
        reject, stats['p-corr'] = multicomp(stats['p-unc'].values, alpha=alpha,
                                       method=padjust)
        stats['p-adjust'] = padjust
        stats['reject'] = reject
    else:
        stats['p-corr'] = None
        stats['p-adjust'] = None
        stats['reject'] = stats['p-unc'] < alpha

    # Trick to force conversion from boolean to int
    stats['reject'] *= 1

    if not return_desc:
        stats[['mean(A)', 'mean(B)', 'std(A)', 'std(B)']] = np.nan

    # Reorganize column order
    col_order = ['Time', 'A', 'B', 'mean(A)', 'std(A)', 'mean(B)', 'std(B)',
                 'Type', 'Paired', 'Alpha', 'T-val', 'Tail', 'p-unc', 'p-corr',
                 'p-adjust', 'reject', 'Eff_size', 'Eff_type']

    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    return stats
