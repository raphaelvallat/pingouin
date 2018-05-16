# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: May 2018
import numpy as np
import pandas as pd
from pingouin import _remove_na

__all__ = ["mwu", "wilcoxon"]


def mwu(x, y, tail='two-sided'):
    """Mann-Whitney U Test (= Wilcoxon rank-sum test). It is the non-parametric
    version of the independent T-test.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'U-val' : U-value
        'p-val' : p-value
        'RBC'   : rank-biserial correlation (effect size)
        'CLES'  : common language effect size

    Notes
    -----
    mwu tests the hypothesis that data in x and y are samples from continuous
    distributions with equal medians. The test assumes that x and y
    are independent. This test corrects for ties and by default
    uses a continuity correction.

    Examples
    --------
    1. Compare the medians of two independent samples.

        >>> import numpy as np
        >>> from pingouin import mwu
        >>> np.random.seed(123)
        >>> x = np.random.uniform(low=0, high=1, size=20)
        >>> y = np.random.uniform(low=0.2, high=1.2, size=20)
        >>> print("Medians = %.2f - %.2f" % (np.median(x), np.median(y)))
        >>> mwu(x, y, tail='two-sided')
            U-val   p-val     RBC    CLES
            97.0    0.006    0.51    0.75
    """
    from scipy.stats import mannwhitneyu
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = _remove_na(x, y, paired=False)
    nx = x.size
    ny = y.size

    # Compute test
    if tail == 'one-sided':
        tail = 'less' if np.median(x) < np.median(y) else 'greater'
    uval, pval = mannwhitneyu(x, y, use_continuity=True, alternative=tail)

    # Effect size 1: common language effect size (McGraw and Wong 1992)
    c = np.array([(a, b) for a in x for b in y])
    num = max((c[:, 0] < c[:, 1]).sum(), (c[:, 0] > c[:, 1]).sum())
    cles = num / (nx * ny)

    # Effect size 2: rank biserial correlation (Wendt 1972)
    rbc = 1 - (2 * uval) / (nx * ny)

    # Fill output DataFrame
    stats = pd.DataFrame({}, index=['MWU'])
    stats['U-val'] = uval.round(3)
    stats['p-val'] = pval
    stats['RBC'] = rbc
    stats['CLES'] = cles

    col_order = ['U-val', 'p-val', 'RBC', 'CLES']
    stats = stats.reindex(columns=col_order)
    return stats


def wilcoxon(x, y, tail='two-sided'):
    """Wilcoxon signed-rank test. It is the non-parametric version of the
    paired T-test.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be related (e.g
        repeated measures).
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'W-val' : W-value
        'p-val' : p-value
        'RBC'   : matched pairs rank-biserial correlation (effect size)
        'CLES'  : common language effect size

    Notes
    -----
    The Wilcoxon signed-rank test tests the null hypothesis that two related
    paired samples come from the same distribution.
    A continuity correction is applied by default.


    Examples
    --------
    1. Compare the medians of two related samples.

        >>> import numpy as np
        >>> from pingouin import wilcoxon
        >>> x = [20, 22, 19, 20, 22, 18, 24, 20]
        >>> y = [38, 37, 33, 29, 14, 12, 20, 22]
        >>> print("Medians = %.2f - %.2f" % (np.median(x), np.median(y)))
        >>> wilcoxon(x, y, tail='two-sided')
    """
    from scipy.stats import wilcoxon
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = _remove_na(x, y, paired=True)
    nx = x.size
    ny = y.size

    # Compute test
    wval, pval = wilcoxon(x, y, zero_method='wilcox', correction=False)
    pval *= .5 if tail == 'one-sided' else pval

    # Effect size 1: common language effect size (McGraw and Wong 1992)
    c = np.array([(a, b) for a in x for b in y])
    num = max((c[:, 0] < c[:, 1]).sum(), (c[:, 0] > c[:, 1]).sum())
    cles = num / (nx * ny)

    # Effect size 2: matched-pairs rank biserial correlation (Kerby 2014)
    rank = np.arange(x.size, 0, -1)
    rsum = rank.sum()
    fav = rank[np.sign(y - x) > 0].sum()
    unfav = rank[np.sign(y - x) < 0].sum()
    rbc = fav / rsum - unfav / rsum

    # Fill output DataFrame
    stats = pd.DataFrame({}, index=['Wilcoxon'])
    stats['W-val'] = wval.round(3)
    stats['p-val'] = pval
    stats['RBC'] = rbc
    stats['CLES'] = cles

    col_order = ['W-val', 'p-val', 'RBC', 'CLES']
    stats = stats.reindex(columns=col_order)
    return stats
