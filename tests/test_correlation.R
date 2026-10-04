library(correlation)

x <- c(4.524991087851508, 4.420531811767872, 3.778970926584845, 12.0, 3.2553302798031223, 4.649569372128418, 2.951786225128683, 4.591357332700219, 1.0489956831264005, 2.9292390843995104, 2.6738018656932527, 4.31118852334745, 5.406716105951538, 3.827585880548361, 4.510669777362404, 5.48020001172895, 5.897501992923205, 3.2481046204119224, 3.6896719823018116, 4.659839225398035, 5.492978998780082, 4.093017621645702, 3.702447504183336, 1.6755435235042087, 2.1236637738250335, 5.622025310958057, 2.7972808778327707, 3.495237872288203, 2.418519025293012, 2.184008274957259)
y <- c(7.417043974095821, 5.073260861980091, 7.2560606695752785, 7.978672340219759, 4.480094096479011, -8.0, 4.380334906909793, 6.202861741341518, 5.004916622004917, 5.274654700615849, 6.007153126165779, 7.362881992968021, 6.836293821056158, 4.549735015009455, 5.7398927665282, 4.977065819368747, 7.271512763937, 5.092800144396517, 6.305237088575375, 6.913523215921814, 5.947704426177541, 6.606245187335942, 5.691865988238505, 4.044863387178897, 6.125520033132931, 6.6929048900606105, 4.083471867309398, 6.451663151111775, 5.988136943566943, 5.140502156598062)
data <- data.frame(x, y)

# Pearson
cor <- cor_test(data, x="x", y="y", method="pearson")
cor <- cor_test(data, x="x", y="y", method="pearson", alternative="less")
cor <- cor_test(data, x="x", y="y", method="pearson", alternative="greater")
# Robust
cor <- cor_test(data, x="x", y="y", method="spearman")
cor <- cor_test(data, x="x", y="y", method="kendall")
cor <- cor_test(data, x="x", y="y", method="percentage")
cor <- cor_test(data, x="x", y="y", method="biweight")
cor <- cor_test(data, x="x", y="y", method="shepherd")

###############################################################################
# Partial correlation
###############################################################################

library(ppcor)

# Update path to pingouin/datasets/
df <- read.csv("../datasets/partial_corr.csv")

# Partial correlation
pcor.test(x=df$x, y=df$y, z=df[, c("cv1")])
pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2")])
pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")], method="spearman")

# Semi partial correlation
# z is removed from the y variable (e.g. y_covar in Pingouin)
spcor.test(x=df$x, y=df$y, z=df$cv1)  # y_covar
spcor.test(x=df$y, y=df$x, z=df$cv1)  # x_covar

spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2")])
spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
spcor.test(x=df$y, y=df$x, z=df[, c("cv1", "cv2", "cv3")])  # x_covar
spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")], method="spearman")


###############################################################################
# Polychoric and tetrachoric correlations
###############################################################################

library(polycor)
library(psych)

# Two-step estimator of polycor, with the standard error and thresholds
show_polychor <- function(name, tab) {
  res <- polychor(tab, ML = FALSE, std.err = TRUE)
  cat(sprintf("%s: rho = %.8f, se = %.8f\n", name, res$rho, sqrt(res$var[1, 1])))
  cat("  row thresholds:", sprintf("%.6f", res$row.cuts), "\n")
  cat("  col thresholds:", sprintf("%.6f", res$col.cuts), "\n")
}

# Contingency tables (rows are filled first, as in NumPy)
t22 <- matrix(c(40, 10, 15, 35), 2, 2, byrow = TRUE)
t33 <- matrix(c(13, 6, 0, 69, 113, 22, 41, 132, 104), 3, 3, byrow = TRUE)  # Olsson 1979
t43 <- matrix(c(131, 71, 20, 217, 207, 112, 213, 337, 257, 52, 139, 244), 4, 3, byrow = TRUE)
t55 <- matrix(c(22, 14, 7, 2, 1, 16, 31, 20, 9, 3, 8, 25, 42, 24, 9, 3, 11, 26, 33, 15,
                1, 4, 10, 17, 21), 5, 5, byrow = TRUE)
# Sparse tables, with several empty cells
t55_sparse <- matrix(c(99, 105, 183, 11, 0, 21, 56, 123, 45, 0, 3, 33, 104, 85, 1,
                       0, 2, 33, 59, 20, 0, 0, 0, 8, 9), 5, 5, byrow = TRUE)
t33_sparse <- matrix(c(5, 2, 0, 3, 6, 1, 0, 2, 4), 3, 3, byrow = TRUE)
t22_sparse <- matrix(c(61661, 85, 1610, 20), 2, 2, byrow = TRUE)
# Negative correlation
t33_neg <- matrix(c(4, 19, 30, 12, 62, 41, 25, 33, 9), 3, 3, byrow = TRUE)
# Empty row and column, which are removed by polychor (with a warning)
t33_empty <- matrix(c(20, 0, 5, 0, 0, 0, 6, 0, 25), 3, 3, byrow = TRUE)

show_polychor("t22", t22)
show_polychor("t33", t33)
show_polychor("t43", t43)
show_polychor("t55", t55)
show_polychor("t55_sparse", t55_sparse)
show_polychor("t33_sparse", t33_sparse)
show_polychor("t22_sparse", t22_sparse)
show_polychor("t33_neg", t33_neg)
show_polychor("t33_empty", t33_empty)

# Boundary: with a perfect association, or an empty cell in a 2x2 table, the likelihood is
# monotone in rho. polychor stops at maxcor = 0.9999 (Pingouin returns -1 or 1).
t33_perfect <- diag(c(10, 20, 30))
t22_zero <- matrix(c(30, 0, 10, 20), 2, 2, byrow = TRUE)
t22_zero_neg <- matrix(c(44268, 14, 193, 0), 2, 2, byrow = TRUE)
polychor(t33_perfect)
polychor(t33_perfect[, 3:1])
polychor(t22_zero)
polychor(t22_zero_neg)

# Tetrachoric correlation with psych. By default, empty cells are replaced by correct = 0.5.
show_tetrachoric <- function(name, tab, correct) {
  res <- tetrachoric(tab, correct = correct)
  cat(sprintf("%s (correct = %s): rho = %.8f, tau = %.6f %.6f\n", name, correct, res$rho,
              res$tau[1], res$tau[2]))
}
show_tetrachoric("t22", t22, 0)
show_tetrachoric("t22_sparse", t22_sparse, 0)
show_tetrachoric("t22_zero", t22_zero, 0.5)
show_tetrachoric("t22_zero_neg", t22_zero_neg, 0.5)
show_tetrachoric("t22_zero_neg2", matrix(c(62503, 768, 105, 0), 2, 2, byrow = TRUE), 0.5)

# Small ordinal datasets. These are generated in Python (see test_polychoric) with:
# rng = np.random.default_rng(seed)
# z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
# x, y = np.digitize(z[:, 0], cuts_x), np.digitize(z[:, 1], cuts_y)

# d1: seed = 1, rho = 0.5, n = 80, cuts_x = [-0.5, 0.5], cuts_y = [-1, 0, 1]
x_d1 <- c(0, 1, 0, 1, 1, 1, 2, 1, 1, 2, 1, 0, 2, 1, 1, 0, 0, 0, 2, 1, 2, 2, 1, 1, 0, 1, 1, 1, 0, 2,
  1, 2, 1, 0, 2, 0, 1, 0, 2, 1, 0, 2, 0, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 2, 0, 2, 1, 1, 1, 2, 0, 2,
  0, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2)
y_d1 <- c(2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 3, 1, 1, 0, 3, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 0, 1, 2,
  2, 2, 1, 2, 3, 0, 3, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 0, 2, 3, 2, 2, 1, 2, 0, 2, 2, 1, 2, 3, 0, 3,
  0, 3, 2, 2, 2, 1, 1, 1, 3, 2, 1, 2, 1, 0, 2, 3, 3, 3)
show_polychor("d1", table(x_d1, y_d1))

# d2: seed = 2, rho = -0.4, n = 60, cuts_x = [0], cuts_y = [-0.3, 0.6]
x_d2 <- c(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0,
  0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0)
y_d2 <- c(1, 0, 2, 1, 1, 2, 0, 1, 1, 0, 2, 1, 0, 2, 0, 2, 2, 1, 1, 0, 2, 0, 1, 1, 0, 1, 2, 2, 2, 0,
  1, 1, 1, 0, 2, 2, 0, 0, 1, 1, 0, 2, 0, 0, 1, 2, 0, 1, 0, 0, 0, 2, 2, 1, 2, 1, 0, 2, 1, 2)
show_polychor("d2", table(x_d2, y_d2))

# d3: seed = 3, rho = 0.7, n = 100, cuts_x = [-1, 0, 1], cuts_y = [-1.2, -0.4, 0.4, 1.2] (with missing values)
x_d3 <- c(1, 1, 2, 3, NA, 1, 2, 3, 1, 1, 1, 1, 1, 0, 1, 2, 0, NA, 3, 3, 1, 2, 1, 0, 1, 2, 1, 2, 2,
  1, NA, 2, 3, 0, 3, 1, 0, 2, 0, 3, 0, 0, 1, 3, 1, 2, 1, 3, 1, 1, 2, 1, 2, 1, 0, 0, 3, 2, NA, 1, 3,
  1, 0, 1, 0, 1, 2, 0, 2, 1, 2, 0, 2, 3, 1, 1, 2, 2, 2, 2, 1, 1, 0, 3, 2, 0, 0, 0, 1, 1, 0, NA, 1,
  2, 1, 2, 2, 3, 1, 1)
y_d3 <- c(0, 1, 2, 4, 4, 2, 2, 3, NA, 1, 3, 1, 2, 0, 3, 3, 1, NA, 4, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1,
  1, 4, 3, 4, 2, 4, 1, 0, 1, 0, 2, 0, 2, 1, 3, 0, 3, 2, 3, 3, 1, 3, 3, 3, 2, 3, 1, 4, 3, 4, 0, 3,
  2, 0, 1, 0, 0, 2, 1, 2, 1, 3, 0, 2, 4, 1, 2, 1, NA, 3, 3, 1, 3, 2, 4, 3, 2, 1, 0, 3, 4, 2, 2, 2,
  3, 3, 3, 2, 2, 2, 2)
show_polychor("d3", table(x_d3, y_d3))
