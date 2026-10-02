# R reference scripts

The scripts in this directory compute the reference values that the Python tests compare
Pingouin against. They are not run by the test suite or by CI: they document where the
hardcoded expected values in the tests come from.

| Script | Python tests | R packages |
| --- | --- | --- |
| `test_correlation.R` | `tests/test_correlation.py` (`test_corr`, `test_partial_corr`) | `correlation`, `ppcor` |

To run a script, install the R packages it loads, then run it from this directory, so that the
relative paths to the datasets (`src/pingouin/datasets/`) resolve:

```bash
cd tests/R
Rscript -e 'install.packages(c("correlation", "ppcor"))'
Rscript test_correlation.R
```
