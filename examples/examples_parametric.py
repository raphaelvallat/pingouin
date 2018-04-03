import numpy as np
from pingouin import gzscore, test_normality, test_homoscedasticity, test_dist

# Generate two random log-normal distributions
np.random.seed(1234)
x = np.random.lognormal(mean=3., sigma=.5, size=200)
y = np.random.lognormal(mean=3., sigma=.6, size=200)

# Test that it comes from a lognormal distribution
from_dist, sig_level = test_dist(x, dist='logistic')
print('Data are from a logistic dist:', from_dist)
print('Significance level (percent): %.2f' % sig_level)

# Compute geometric z-score
gx, gy = gzscore(x), gzscore(y)

# Test for normality
norm, _ = test_normality(gx, gy)
print('Normality:', norm)

# Test for equality of variances (homoscedasticity)
equal_var, _ = test_homoscedasticity(gx, gy)
print('Equal variances:', equal_var)
