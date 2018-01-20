# -*- coding: utf-8 -*-
import numpy as np
from problem import random_problem, Settings

eps = np.finfo(np.float32).eps # global eps

# Moore-Penrose inverse of a diagonal matrix
# Both input and output are represented as matrices.
# WARNING: This function assumes PSD input!
# WARNING: This function changes the input! (in-place)
def dagger(v):
    mask = v > eps
    v[mask] = np.reciprocal(v[mask])
    return v

# solves the problem
# min_W 1/2 * ||Y - XW||^2 + alpha * ||W||_2^2
def ridge(X, Y, alpha):
    return np.linalg.inv(X.T.dot(X) + alpha * np.identity(X.shape[1])).dot(X.T).dot(Y)

def solve(Y, H, iterations, mu):
    N, K = Y.shape
    M = H.shape[1]
    V = np.identity(M) / M
    sqrtV = np.sqrt(V)
    for i in range(iterations):
        ### update W ###
        X = H.dot(sqrtV)
        W = sqrtV.dot(ridge(X, Y, mu))
        ### update V ###
        W2 = np.sum(W**2, axis=1)
        V = np.diag(W2 / np.sum(W2))
        sqrtV = np.sqrt(V)
        print V
    print Y
    return W