# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from pingouin.utils import _check_eftype, _remove_na
from pingouin.parametric import test_homoscedasticity


__all__ = ["compute_esci", "convert_effsize", "compute_effsize",
           "compute_effsize_from_t"]


def compute_esci(x=None, y=None, ef=None, nx=None, ny=None, alpha=.95,
                 method='parametric', n_boot=2000, eftype='cohen',
                 return_dist=False):
    """Bootstrapped or parametric confidence intervals of an effect size.

    Parameters
    ----------
    x, y : int
        Data vectors (required for bootstrapping only)
    ef : float
        Original effect size (must be Cohen d or Hedges g).
        Required for 'parametric' method only.
    nx, ny : int
        Length of vector x and y.
    alpha : float, optional
        Confidence interval (0.95 = 95%)
    method : string
        Computation method
        Available methods are ::

        'parametric' : uses standard deviation of effect sizes.
        'bootstrap' : uses a bootstrapping procedure (pivotal CI).
    n_boot : int
        Number of permutations for the bootstrap procedure
    eftype : string
        Effect size type for the bootstrapping procedure.
    return_dist : boolean
        If True, return the distribution of permutations (e.g. useful for a
        posteriori plotting)

    Returns
    -------
    ci : array
        Desired converted effect size

    Examples
    --------
    1. Compute the 95% **parametric** confidence interval of an effect size
       given the two sample sizes.

        >>> import numpy as np
        >>> from pingouin import compute_esci, compute_effsize
        >>> np.random.seed(123)
        >>> x = np.random.normal(loc=3, size=60)
        >>> y = np.random.normal(loc=2, size=50)
        >>> ef = compute_effsize(x=x, y=y, eftype='cohen')
        >>> print(ef)
        >>> print(compute_esci(ef=ef, nx=len(x), ny=len(y)))
            1.01
            [0.61  1.41]


    2. Compute the 95% **bootstrapped** confidence interval of an effect size.
       In that case, we need to pass directly the original x and y arrays.

        >>> print(compute_esci(x=x, y=y, method='bootstrap'))
            [0.93 1.17]


    3. Plot the bootstrapped distribution using Seaborn.

        >>> import seaborn as sns
        >>> ci, dist = compute_esci(x=x, y=y, method='bootstrap',
        >>>                         return_dist=True, n_boot=5000)
        >>> sns.distplot(dist)


    4. Get the 68% confidence interval

        >>> ci68 = compute_esci(x=x, y=y, method='bootstrap', alpha=.68)
        >>> print(ci68)
            [0.99 1.12]


    5. Compute the bootstrapped Pearson r confidence interval

        >>> ef = compute_effsize(x=x, y=y, eftype='r')
        >>> ci = compute_esci(x=x, y=y, method='bootstrap', eftype='r')
        >>> print(ef)
        >>> print(ci)
            0.45
            [0.42 0.51]
    """
    # Check arguments
    if not _check_eftype(eftype):
        err = "Could not interpret input '{}'".format(eftype)
        raise ValueError(err)
    if all(v is None for v in [x, y, nx, ny]):
        raise ValueError("You must either specify x and y or nx and ny")
    if x is None and y is None and method == 'bootstrap':
        method = 'parametric'
    if nx is None and ny is None and x is not None and y is not None:
        nx = len(x)
        ny = len(y)

    # Start computation
    if method == 'parametric':
        from scipy.stats import norm
        se = np.sqrt(((nx + ny) / (nx * ny)) + (ef**2) / (2 * (nx + ny)))
        crit = np.abs(norm.ppf((1 - alpha) / 2))
        ci = np.array([ef - crit * se, ef + crit * se])
        return ci
    elif method == 'bootstrap':
        ef = compute_effsize(x=x, y=y, eftype=eftype)
        rd_x = np.random.choice(nx, size=n_boot)
        rd_y = np.random.choice(ny, size=n_boot)
        effsizes = np.zeros(n_boot)

        for i in np.arange(n_boot):
            x_new = x.copy()
            y_new = y.copy()
            x_new[rd_x[i]] = y[rd_y[i]]
            y_new[rd_y[i]] = x[rd_x[i]]
            effsizes[i] = compute_effsize(x=x_new, y=y_new, eftype=eftype)

        ef_sorted = np.sort(effsizes)
        lower = int(n_boot * ((1 - alpha) / 2))
        upper = int(n_boot * (alpha + (1 - alpha) / 2))
        ci = np.array([ef_sorted[lower], ef_sorted[upper]])
        # Pivot confidence intervals
        ci = np.sort(2 * ef - ci)
        if return_dist:
            return ci, effsizes
        else:
            return ci


def convert_effsize(ef, input_type, output_type, nx=None, ny=None):
    """Conversion between effect sizes.

    Parameters
    ----------
    ef : float
        Original effect size
    input_type : string
        Effect size type of ef. Must be 'r' or 'd'.
    output_type : string
        Desired effect size type.
        Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
    nx, ny : int, optional
        Length of vector x and y.
        nx and ny are required to convert to Hedges g

    Returns
    -------
    ef : float
        Desired converted effect size

    See Also
    --------
    compute_effsize : Calculate effect size between two set of observations.
    compute_effsize_from_t : Convert a T-statistic to an effect size.

    Examples
    --------
    1. Convert from Cohen d to eta-square

        >>> from pingouin import convert_effsize
        >>> d = .45
        >>> eta = convert_effsize(d, 'cohen', 'eta-square')
        >>> print(eta)
            0.05


    2. Convert from Cohen d to Hegdes g (requires the sample sizes of each
    group)

        >>> d = .45
        >>> g = convert_effsize(d, 'cohen', 'hedges', nx=10, ny=10)
        >>> print(eta)
            0.43


    3. Convert Pearson r to Cohen d

        >>> r = 0.40
        >>> d = convert_effsize(r, 'r', 'cohen')
        >>> print(d)
            0.87


    4. Reverse operation: convert Cohen d to Pearson r

        >>> d = 0.873
        >>> r = convert_effsize(d, 'cohen', 'r')
        >>> print(r)
            0.40
    """
    it = input_type.lower()
    ot = output_type.lower()

    # Check input and output type
    for input in [it, ot]:
        if not _check_eftype(input):
            err = "Could not interpret input '{}'".format(input)
            raise ValueError(err)
    if it not in ['r', 'cohen']:
        raise ValueError("Input type must be 'r' or 'cohen'")

    if it == ot:
        return ef

    if it == 'r':
        # Rosenthal 1994
        d = (2 * ef) / np.sqrt(1 - ef**2)
    elif it == 'cohen':
        d = ef

    # Then convert to the desired output type
    if ot == 'cohen':
        return d
    elif ot == 'hedges':
        if all(v is not None for v in [nx, ny]):
            return d * (1 - (3 / (4 * (nx + ny) - 9)))
        else:
            # If shapes of x and y are not known, return cohen's d
            print("You need to pass nx and ny arguments to compute Hedges g.",
                  "Returning Cohen's d instead")
            return d
    elif ot == 'glass':
        print("Returning original effect size instead of Glass because",
              "variance is not known.")
        return ef
    elif ot == 'r':
        # McGrath and Meyer 2006
        if all(v is not None for v in [nx, ny]):
            a = ((nx + ny)**2 - 2 * (nx + ny)) / (nx * ny)
        else:
            a = 4
        return d / np.sqrt(d**2 + a)
    elif ot == 'eta-square':
        # Cohen 1988
        return (d / 2)**2 / (1 + (d / 2)**2)
    elif ot == 'odds-ratio':
        # Borenstein et al. 2009
        return np.exp(d * np.pi / np.sqrt(3))
    elif ot == 'auc':
        # Ruscio 2008
        from scipy.stats import norm
        return norm.cdf(d / np.sqrt(2))
    elif ot == 'none':
        return None


def compute_effsize(x, y, paired=False, eftype='cohen'):
    """Calculate effect size between two set of observations.

    Parameters
    ----------
    x : np.array or list
        First set of observations.
    y : np.array or list
        Second set of observations.
    paired : boolean
        If True, uses Cohen d-avg formula to correct for repeated measurements
        (Cumming 2012)
    eftype : string
        Desired output effect size.
        Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve

    Returns
    -------
    ef : float
        Effect size

    See Also
    --------
    convert_effsize : Conversion between effect sizes.
    compute_effsize_from_t : Convert a T-statistic to an effect size.

    Examples
    --------
    1. Compute Cohen d from two independent set of observations.

        >>> import numpy as np
        >>> from pingouin import compute_effsize
        >>> np.random.seed(123)
        >>> x = np.random.normal(2, size=100)
        >>> y = np.random.normal(2.3, size=95)
        >>> d = compute_effsize(x=x, y=y, eftype='cohen', paired=False)
        >>> print(d)
            -0.28

    2. Compute Hedges g from two paired set of observations.

        >>> import numpy as np
        >>> from pingouin import compute_effsize
        >>> np.random.seed(123)
        >>> x = [1.62, 2.21, 3.79, 1.66, 1.86, 1.87, 4.51, 4.49, 3.3 , 2.69]
        >>> y = [0.91, 3., 2.28, 0.49, 1.42, 3.65, -0.43, 1.57, 3.27, 1.13]
        >>> g = compute_effsize(x=x, y=y, eftype='hedges', paired=True)
        >>> print(g)
            0.88

    3. Compute Glass delta from two independant set of observations. The group
       with the lowest variance will automatically be selected as the control.

        >>> import numpy as np
        >>> from pingouin import compute_effsize
        >>> np.random.seed(123)
        >>> x = np.random.normal(2, scale=1, size=50)
        >>> y = np.random.normal(2, scale=2, size=45)
        >>> d = compute_effsize(x=x, y=y, eftype='glass')
        >>> print(d)
            -0.12
    """
    # Check arguments
    if not _check_eftype(eftype):
        err = "Could not interpret input '{}'".format(eftype)
        raise ValueError(err)

    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size and paired:
        print('x and y have unequal sizes. Switching to paired == False.')
        paired = False

    # Remove NA
    x, y = _remove_na(x, y, paired=paired)
    nx = x.size
    ny = y.size

    if ny == 1:
        # Case 1: One-sample Test
        d = (np.mean(x) - y) / np.std(x)
        return d

    if eftype.lower() == 'glass':
        # Find group with lowest variance
        sd_control = np.min([np.std(x), np.std(y)])
        d = (np.mean(x) - np.mean(y)) / sd_control
        return d
    else:
        # Test equality of variance of data with a stringent threshold
        equal_var, p = test_homoscedasticity(x, y, alpha=.001)
        if not equal_var:
            print('Unequal variances (p<.001). You should consider reporting',
                  'Glass delta instead.')

        # Compute unbiased Cohen's d effect size
        if not paired:
            # https://en.wikipedia.org/wiki/Effect_size
            dof = nx + ny - 2
            poolsd = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 +
                              (ny - 1) * np.std(y, ddof=1)**2) / dof)
            d = (np.mean(x) - np.mean(y)) / poolsd
        else:
            # Report Cohen d-avg (Cumming 2012; Lakens 2013)
            d = (np.mean(x) - np.mean(y)) / (.5 * (np.std(x) + np.std(y)))
        return convert_effsize(d, 'cohen', eftype, nx=nx, ny=ny)


def compute_effsize_from_t(tval, nx=None, ny=None, N=None, eftype='cohen'):
    """Compute effect size from a T-value.

    Parameters
    ----------
    tval : float
        T-value
    nx, ny : int, optional
        Group sample sizes.
    N : int, optional
        Total sample size (will not be used if nx and ny are specified)
    eftype : string, optional
        desired output effect size

    Returns
    -------
    ef : float
        Effect size

    See Also
    --------
    compute_effsize : Calculate effect size between two set of observations.
    convert_effsize : Conversion between effect sizes.

    Examples
    --------
    1. Compute effect size from a T-value when both sample sizes are known.

        >>> from pingouin import compute_effsize_from_t
        >>> tval, nx, ny = 2.90, 35, 25
        >>> d = compute_effsize_from_t(tval, nx=nx, ny=ny, eftype='cohen')
        >>> print(d)
            0.76

    2. Compute effect size when only total sample size is known (nx+ny)

        >>> tval, N = 2.90, 60
        >>> d = compute_effsize_from_t(tval, N=N, eftype='cohen')
        >>> print(d)
            0.75
    """
    if not _check_eftype(eftype):
        err = "Could not interpret input '{}'".format(eftype)
        raise ValueError(err)

    if not isinstance(tval, float):
        err = "T-value must be float"
        raise ValueError(err)

    # Compute Cohen d (Lakens, 2013)
    if nx is not None and ny is not None:
        d = tval * np.sqrt(1 / nx + 1 / ny)
    elif N is not None:
        d = 2 * tval / np.sqrt(N)
    else:
        raise ValueError('You must specify either nx + ny, or just N')
    return convert_effsize(d, 'cohen', eftype, nx=nx, ny=ny)
