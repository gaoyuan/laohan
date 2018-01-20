# -*- coding: utf-8 -*-
import numpy as np
from problem import random_problem, Settings
from sklearn.linear_model import Lasso

def solve(Y, H, mu, iterations, epsilon):
    N, K = Y.shape
    M = H.shape[1]
    alpha = np.linalg.norm(Y, axis=0)
    clf = Lasso(alpha=mu)
    traces = [] # aggregate running info for analysis purposes
    
    for i in range(iterations):
        alpha_prev = alpha

        ### update phi ###
        Z = Y / alpha
        flattened_Z  = Z.flatten('F')
        augmented_H = np.tile(H, (K, 1))
        clf.fit(augmented_H, flattened_Z)
        phi = clf.coef_
        phi /= np.linalg.norm(phi) # normalize phi to resolve multiplicative factor
        ### update alpha ###
        m = H.dot(phi)
        alpha = m.T.dot(Y) / (m.T.dot(m))

        obj = 1/2 * np.linalg.norm(Y - np.outer(m, alpha))**2 + mu * np.linalg.norm(phi, 1)
        traces.append([obj, phi, alpha])

        # convergence check
        if np.linalg.norm(alpha_prev - alpha) < epsilon:
            print "Converged in %d iterations." % (i + 1)
            return phi, alpha, traces

    print "Does not converge in %d iterations." % iterations
    return phi, alpha, traces