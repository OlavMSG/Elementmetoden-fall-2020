# -*- coding: utf-8 -*-
"""
Created on 10.10.2020

@author: Olav Milian
"""

import numpy as np
from Gauss_quadrature import Gauss_quadrature_tester
from contour_and_mesh_plotter import meshplot
from Poisson2D import Dirichlet2D, Mixed2D

"""The main script"""

Gauss_quadrature_tester()

# save the plots?
save = False
# Do the Singularity check?
DoSingularityCheck = True
# list of N to plot for
N_list = [100, 500, 1000]

# make a meshplot
meshplot(N_list, save=save)

# source function
f = lambda x, y: 16 * np.pi * np.pi * (x * x + y * y) * np.sin(2 * np.pi * (x * x + y * y)) \
                 - 8 * np.pi * np.cos(2 * np.pi * (x * x + y * y))
# Neumann B.C. function on x^2 + y^2 = r^2 = 1
# 4 * np.pi  * np.sqrt(x * x + y * y) * np.cos(2 * np.pi * (x * x + y * y)) = 4 * np.pi on x^2 + y^2 = r^2 = 1
g = lambda x, y: 4 * np.pi
# exact solution
u_exact = lambda x, y: np.sin(2 * np.pi * (x * x + y * y))

for N in N_list:
    Dirichlet2D(N, f, u_exact, epsilon=1e-6, DoSingularityCheck=DoSingularityCheck, save=save)
    Mixed2D(N, f, g, u_exact, epsilon=1e-6, DoSingularityCheck=DoSingularityCheck, save=save)

