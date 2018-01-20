# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from problem import random_problem, Settings
from em_vec import solve
H, phi, alpha, W, Y = random_problem(Settings())
p, a, traces = solve(Y, H, 0.01, 100, 0.0001)
print phi, p
print alpha, a