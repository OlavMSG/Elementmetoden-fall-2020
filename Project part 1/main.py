# -*- coding: utf-8 -*-
"""
Created on 10.10.2020

@author: Olav Gran
based in old code by Ruben Mustad
"""

import numpy as np
from Gauss_quadrature import Gauss_quadrature_tester
from contour_and_mesh_plotter import meshplot
from Poisson2D import Dirichlet2D, Mixed2D

"""The main script"""

# Parameters
# ----------
# N_list : list of ints
#     list of different N-s.
# N : int
#     Number of nodes in the mesh.
# f : function pointer
#     pointer to function to be equal to on the right hand side.
# u_exact : function pointer
#     Pointer to the function for the exact solution.
# g_N : function pointer
#     pointer to function for the Neumann B.C.
# g_D : function pointer
#     pointer to function for the Dirichlet B.C.
# DoSingularityCheck : bool, optional
#     Call the singularity_check function to check i A is singular before implementation of B.C. The default is True.
# save : bool, optional
#     Will the plot be saved in the plot folder. The default is False.
#     Note: the plot folder must exist!


# list of N to plot for
N_list = [100, 500, 1000]

# source function
f = lambda x, y: 16 * np.pi * np.pi * (x * x + y * y) * np.sin(2 * np.pi * (x * x + y * y)) \
                      - 8 * np.pi * np.cos(2 * np.pi * (x * x + y * y))
# exact solution
u_exact = lambda x, y:  np.sin(2 * np.pi * (x * x + y * y))

# Neumann B.C. function on x^2 + y^2 = r^2 = 1, y>0
# 4 * np.pi  * np.sqrt(x * x + y * y) * np.cos(2 * np.pi * (x * x + y * y)) = 4 * np.pi on x^2 + y^2 = r^2 = 1
g_N = lambda x, y: 4 * np.pi

# Dirichlet B.C function on x^2 + y^2 = r^2 = 1 or x^2 + y^2 = r^2 = 1, y<=0
g_D = lambda x, y: np.zeros_like(x)

# Do the Singularity check?
DoSingularityCheck = True
# save the plots?
save = False

# test the Gauss Quadratue script
Gauss_quadrature_tester()

# make a meshplot
meshplot(N_list, save=save)

# Do the computations and plotting for all N in N_list
for N in N_list:
    # do Dirichlet BC
    Dirichlet2D(N, f, u_exact, g_D, DoSingularityCheck=DoSingularityCheck, save=save)
    # do Mixed BC
    Mixed2D(N, f, u_exact, g_N, g_D, DoSingularityCheck=DoSingularityCheck, save=save)

