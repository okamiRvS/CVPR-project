from scipy.linalg import null_space
import numpy as np


def compute(x_i, X_i):
    x = np.zeros((2, 12))
    x[0, 4:8] = X_i * -1
    x[0, 8:12] = x_i[1] * X_i
    x[1, 0:4] = X_i
    x[1, 8:12] = -x_i[0] * X_i
    return x

x = np.array([
    [360, 48, 1],
    [920, 48, 1],
    [304, 277, 1],
    [975, 277, 1],
    [217, 630, 1],
    [1059, 630, 1],
])

X = np.array([
    [-8.89, 1.7845, 0.05, 1],
    [8.89, 1.7845, 0, 1],
    [-8.89, 0, 0, 1],
    [8.89, 0, 0, 1],
    [-8.89, -1.7845, 0, 1],
    [8.89, -1.7845, 0, 1],
])

A = None
for x_i, X_i in zip(x, X):
    A = compute(x_i, X_i) if A is None else np.concatenate((A, compute(x_i, X_i)))

p = null_space(A)
p = p.reshape((3, 4))
