import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from pingouin import rcorr


@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("adjust", [None, "bonferroni", "sidak", "holm", "fdr_bh", "fdr_by"])
@pytest.mark.parametrize("missing", [False, True])
def test_rcorr_adjusts_only_unique_pairs(method, adjust, missing):
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(80, 5)))
    if missing:
        frame.iloc[:5, 0] = np.nan
        frame.iloc[10:15, 2] = np.nan
    i, j = np.triu_indices(frame.shape[1], 1)
    correlation = pearsonr if method == "pearson" else spearmanr
    raw = []
    for a, b in zip(i, j):
        pairs = frame.iloc[:, [a, b]].dropna()
        raw.append(correlation(pairs.iloc[:, 0], pairs.iloc[:, 1]).pvalue)
    expected = np.asarray(raw) if adjust is None else multipletests(raw, method=adjust)[1]
    actual = rcorr(frame, method=method, padjust=adjust, stars=False, decimals=12)
    np.testing.assert_allclose(actual.to_numpy()[i, j].astype(float), expected, atol=1e-12)


def test_rcorr_fdr_stars_do_not_include_placeholder_tests():
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(100, 20)))
    i, j = np.triu_indices(20, 1)
    actual = rcorr(frame, padjust="fdr_bh")
    assert all(value == "" for value in actual.to_numpy()[i, j])
