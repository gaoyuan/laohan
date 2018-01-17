# -*- coding: utf-8 -*-
import numpy as np
from problem import random_problem, Settings
from sklearn.linear_model import Lasso

def solve(Y, H, iterations, mu):
    N, K = Y.shape
    M = H.shape[1]
    alpha = np.linalg.norm(Y, axis=0)
    clf = Lasso(alpha=mu)
    
    for i in range(iterations):
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

    return phi, alpha


H, phi, alpha, W, Y = random_problem(Settings(N = 3, M = 3, K = 3, eta = 0.01, phi_density = 0.8, seed = 0))
print phi, alpha
print phi.dot(alpha.T)
p, a = solve(Y, H, 100, 0.001)
print p, a
print np.reshape(p, (3,1)).dot(np.reshape(alpha, (1, 3)))