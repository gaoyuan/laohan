# -*- coding: utf-8 -*-
import pytest
import solver
import numpy as np


def test_dagger():
    v = np.array([0.0, 0.2, 0.0, 2.0, 0.0])
    v_dagger = np.array([0.0, 5.0, 0.0, 0.5, 0.0])
    np.testing.assert_array_almost_equal(solver.dagger(v), v_dagger)
    np.testing.assert_array_almost_equal(v, v_dagger) # make sure it is in-place modification

def test_ridge_simple():
    X = np.identity(2)
    Y = np.random.rand(2, 2)
    alpha = 0
    W = Y
    np.testing.assert_array_almost_equal(solver.ridge(X, Y, alpha), W)

def test_ridge_sklearn():
    X = np.random.rand(3, 5)
    Y = np.random.rand(3, 2)
    alpha = 0.01
    from sklearn.linear_model import Ridge
    model = Ridge(alpha = alpha, tol=0.1, fit_intercept=False).fit(X, Y)
    np.testing.assert_array_almost_equal(solver.ridge(X, Y, alpha), model.coef_.T)
