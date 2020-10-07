# -*- coding: utf-8 -*-
"""
Created on 07.10.2020

@author: Olav Milian
"""
import numpy as np
import scipy.sparse as sparse
from getdisc import GetDisc
from Gauss_quadrature import quadrature2d
from contour_and_mesh_plotter import contourplot


def Dirichlet(N, f, epsilon=1e-6):
    # solve Au = F using Dirichlet b.c.
    p, tri, edge = GetDisc(N)
    A = sparse.lil_matrix((N, N))
    F = np.zeros(N)
    F_local = np.zeros(3)

    for k in range(len(tri)):
        # calculate basis functions, gradient of jacobian etc.
        # node-numbers for the k'th triangle
        nk = tri[k, :]
        p1, p2, p3 = p[nk[0], :], p[nk[1], :], p[nk[2], :]
        # row: [1, Cx, Cy]
        Mk = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, p3[0], p3[1]]])
        Ck = np.linalg.solve(Mk, np.eye(3))
        # phi_1 = [1, x, y] @ Ck[:,0]
        # phi_2 = [1, x, y] @ Ck[:,1]
        # phi_3 = [1, x, y] @ Ck[:,2]
        phi = lambda x, y: [1, x, y] @ Ck

        # Calculate the local stifness matrix Ak
        # Ak_{a,b} = Area_k * (Cxk_a * Cx_b + Cyk_a * Cyk_b)
        Area_k_func = lambda x, y: 1
        Area_k = quadrature2d(p1, p2, p3, 4, Area_k_func)
        # Ck[1:3, :].T = [[Cxk_1, Cyk_1], [Cxk_2, Cy_2], [Cxk_3, Cyk_3]]
        # Ck[1:3, :] = [[Cxk_1, Cxk_2, Cxk_3], [Cyk_1, Cyk_2, Cyk_3]]
        # (Ck[1:3, :].T @ Ck[1:3, :])_{a, b} = Cxk_a * Cxk_b + Cyk_a * Cyk_b, (a,b) in {1, 2, 3}
        Ak = Area_k * Ck[1:3, :].T @ Ck[1:3, :]

        # Add the Ak to the global stiffness matrix A
        A[np.ix_(tri[k, :], tri[k, :])] += Ak

        # Calculate the local load vector
        for i in range(3):
            f_phi_i = lambda x, y: f(x, y) * phi(x, y)[i]
            F_local[i] = quadrature2d(p1, p2, p3, 4, f_phi_i)
        # Add the local load vector Fk to the global load vector F
        F[np.ix_(tri[k, :])] += F_local
    print('The condition number of matrix A is: ' + str(np.linalg.cond(A.toarray())))

    # Dirichlet boundary conditions
    print('The max value of the stiffness matrix A is ' + str(np.max(A.toarray())))
    for k in range(0, N - 1):
        if p[k, 0] ** 2 + p[k, 1] ** 2 + epsilon > 1:
            F[k] = 0
            F[-1] = 0
            A[k, k] = 1 / epsilon
            A[-1, -1] = 1 / epsilon
    return sparse.linalg.spsolve(A.tocsr(), F)

if __name__ == "__main__":
    # source function
    f = lambda x, y: 16 * np.pi ** 2 * (x ** 2 + y ** 2) * np.sin(2 * np.pi * (x ** 2 + y ** 2)) \
                          - 8 * np.pi * np.cos(2 * np.pi * (x ** 2 + y ** 2))
    u_exact = lambda x, y:  np.sin(2 * np.pi * (x * x + y * y))
    N = 2000
    U_dir = Dirichlet(N, f)
    # Create plots
    BC_type = 'Dirichlet'
    contourplot(N, U_dir, BC_type, u_exact)


