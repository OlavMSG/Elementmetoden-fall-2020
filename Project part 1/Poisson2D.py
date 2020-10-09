# -*- coding: utf-8 -*-
"""
Created on 07.10.2020

@author: Olav Milian
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.special import roots_legendre
from getdisc import GetDisc
from Gauss_quadrature import lineintegral, quadrature2D
from contour_and_mesh_plotter import contourplot


def Base_Poisson2D(p, tri, f, DoSingularityCheck):
    # Stiffness matrix
    A = sparse.lil_matrix((N, N))
    # load vector
    F = np.zeros(N)

    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        p1 = p[nk[0], :]
        p2 = p[nk[1], :]
        p3 = p[nk[2], :]
        # calculate basis functions, gradient of jacobian etc.
        # row_k: [1, x_k, y_k]
        Mk = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, p3[0], p3[1]]])
        Ck = np.linalg.inv(Mk)  # here faster than solving Mk @ Ck = I_3
        # phi_1 = [1, x, y] @ Ck[:,0]
        # phi_2 = [1, x, y] @ Ck[:,1]
        # phi_3 = [1, x, y] @ Ck[:,2]
        phi = lambda x, y: [1, x, y] @ Ck

        # Calculate the local stifness matrix Ak
        # Ak_{a,b} = Area_k * (Cxk_a * Cx_b + Cyk_a * Cyk_b)
        Area_k_func = lambda x, y: 1
        Area_k = quadrature2D(p1, p2, p3, 4, Area_k_func)
        # Ck[1:3, :].T = [[Cxk_1, Cyk_1], [Cxk_2, Cy_2], [Cxk_3, Cyk_3]]
        # Ck[1:3, :] = [[Cxk_1, Cxk_2, Cxk_3], [Cyk_1, Cyk_2, Cyk_3]]
        # (Ck[1:3, :].T @ Ck[1:3, :])_{a, b} = Cxk_a * Cxk_b + Cyk_a * Cyk_b, (a,b) in {1, 2, 3}
        Ak = Area_k * Ck[1:3, :].T @ Ck[1:3, :]
        # Add the Ak to the global stiffness matrix A
        A[np.ix_(nk, nk)] += Ak

        # Calculate the local load vector
        F_local = np.zeros(3)
        for i in range(3):
            f_phi_i = lambda x, y: f(x, y) * phi(x, y)[i]
            F_local[i] = quadrature2D(p1, p2, p3, 4, f_phi_i)
        # Add the local load vector Fk to the global load vector F
        F[nk] += F_local

    if DoSingularityCheck:
        # check condition number
        condA = np.linalg.cond(A.toarray())
        print('The condition number of matrix A is: ' + str(condA))
        # Check the max value of A
        maxA = np.max(A.toarray())
        print('The max value of the stiffness matrix A is ' + str(maxA))
        # if the condition number is larger than 1/eps vere eps is the machine epsilon, then A is most likely singular
        if condA > 1 / np.finfo(A.dtype).eps:
            print("A is most likely singular before implementation of BC.")

    return A, F


def Dirichlet(N, f, epsilon=1e-6, DoSingularityCheck=True):
    # solve Au = F using Dirichlet b.c.
    p, tri, edge = GetDisc(N)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.

    # get the stiffness matrix and load vector
    A, F = Base_Poisson2D(p, tri, f, DoSingularityCheck)

    # Dirichlet boundary conditions
    for ek in edge:
        A[np.ix_(ek, ek)] = 1 / epsilon
        F[ek] = 0

    return spsolve(A.tocsr(), F)


def Split_edge_nodes(p, edge):
    diri = []
    neu = []
    both = []

    for ek in edge:
        p1 = p[ek[0]]
        p2 = p[ek[1]]
        if p1[1] <= 0 and p2[1] <= 0:
            diri.append(ek)
        elif p1[1] > 0 and p2[1] > 0:
            neu.append(ek)
        else:
            both.append(ek)

    neu = np.asarray(neu)
    diri = np.asarray(diri)
    both = np.asarray(both)

    return neu, diri, both


def Mixed(N, f, g, epsilon=1e-6, DoSingularityCheck=True):
    # solve Au = F using Dirichlet b.c.
    p, tri, edge = GetDisc(N)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.

    # get the stiffness matrix and load vector
    A, F = Base_Poisson2D(p, tri, f, DoSingularityCheck)

    neu, diri, both = Split_edge_nodes(p, edge)
    # Dirichlet boundary condition
    for ek in diri:
        A[np.ix_(ek, ek)] = 1 / epsilon
        F[ek] = 0

    # Neumann boundary conditions
    for ek in neu:
        p1 = p[ek[0]]
        p2 = p[ek[1]]
        F[ek] += lineintegral(p1, p2, 4, g)

    # chose to use Dirichlet
    for ek in both:
        A[np.ix_(ek, ek)] = 1 / epsilon
        F[ek] = 0

    return spsolve(A.tocsr(), F)

if __name__ == "__main__":
    # source function
    f = lambda x, y: 16 * np.pi ** 2 * (x ** 2 + y ** 2) * np.sin(2 * np.pi * (x ** 2 + y ** 2)) \
                          - 8 * np.pi * np.cos(2 * np.pi * (x ** 2 + y ** 2))
    # Neumann B.C. function
    g = lambda x, y: 4 * np.pi * np.sqrt(x ** 2 + y ** 2) * np.cos(2 * np.pi * (x ** 2 + y ** 2))
    # exact solution
    u_exact = lambda x, y:  np.sin(2 * np.pi * (x * x + y * y))

    # save the plots?
    save = False
    # chose the N
    N = 1000
    # get numerical solution
    U_dir = Dirichlet(N, f)
    # Create plot
    BC_type = 'Dirichlet'
    contourplot(N, U_dir, BC_type, u_exact, save=save)
    print("-"*40)
    # get numerical solution
    U_mix = Mixed(N, f, g)
    # Create plot
    BC_type = "Mixed"
    contourplot(N, U_mix, BC_type, u_exact, save=save)





