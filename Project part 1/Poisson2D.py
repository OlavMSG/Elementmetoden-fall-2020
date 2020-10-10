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
from Gauss_quadrature import quadrature2D
from contour_and_mesh_plotter import contourplot


def singularity_check(A):
    # check condition number
    condA = np.linalg.cond(A.toarray())
    print("-" * 60)
    print('The condition number of matrix A is: ' + str(condA))
    # Check the max value of A
    maxA = np.max(A.toarray())
    print('The max value of the stiffness matrix A is ' + str(maxA))
    # if the condition number is larger than 1/eps vere eps is the machine epsilon, then A is most likely singular
    if condA > 1 / np.finfo(A.dtype).eps:
        print("A is most likely singular before implementation of BC.")
    print("-" * 60)


def lineintegral(a, b, Nq, g):
    # function to handle the lineintegral with local basis functions implemented
    # Weights and gaussian quadrature points
    z_q, rho_q = roots_legendre(Nq)
    a = np.asarray(a)
    b = np.asarray(b)
    # parameterization of the line C between a and b,
    # r(t) = (1-t)a/2 + (1+t)b/2,
    x = lambda t: ((1 - t) * a[0] + (1 + t) * b[0]) / 2
    y = lambda t: ((1 - t) * a[1] + (1 + t) * b[1]) / 2
    # r'(t) = -a/2+ b/2 = (b-a)/2, |r'(t)| = norm(b-a)
    abs_r_t = np.linalg.norm(b - a, ord=2) / 2
    # g times local basis
    g1 = lambda t: g(x(t), y(t)) * (1 + t) / 2  # for a
    g2 = lambda t: g(x(t), y(t)) * (1 - t) / 2  # for b
    # int_C g(x, y) * phi(x, y) ds = int_{-1}^1 g(r(t)) * phi(r(t)) |r'(t)| * 2 dt
    # = norm(b-a)  int_{-1}^1 g(r(t)) * phi(r(t)) dt
    I1 = abs_r_t * np.sum(rho_q * g1(z_q))  # load for a
    I2 = abs_r_t * np.sum(rho_q * g2(z_q))  # load for b
    return I1, I2


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


def Base_Poisson2D(N, p, tri, f, DoSingularityCheck=True):
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
        singularity_check(A)

    return A, F


def Dirichlet_Poisson2D(N, f, epsilon=1e-6, DoSingularityCheck=True):
    # solve Au = F using Dirichlet b.c.
    p, tri, edge = GetDisc(N)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.

    # get the stiffness matrix and load vector
    A, F = Base_Poisson2D(N, p, tri, f, DoSingularityCheck)

    # Dirichlet boundary conditions
    for ek in edge:
        A[np.ix_(ek, ek)] = 1 / epsilon
        F[ek] = 0

    return spsolve(A.tocsr(), F)


def Mixed_Poisson2D(N, f, g, epsilon=1e-6, DoSingularityCheck=True):
    # solve Au = F using Dirichlet b.c.
    p, tri, edge = GetDisc(N)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.

    # get the stiffness matrix and load vector
    A, F = Base_Poisson2D(N, p, tri, f, DoSingularityCheck)

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

    for ek in both:
        # p1 = (x1, y1), p2=(x2, y2)
        p1 = p[ek[0]]
        p2 = p[ek[1]]
        # Here either y1 > 0 and y2 < 0, or y1 < 0 and y2 > 0
        # Parameterize the line as r(t)= (x(t), y(t)) = (1 - t ) * p1 - t * p2, t in [0, 1]
        # Let t0 be solution to 0 = y(t) = (1 - t) * y1 + t * y2
        t0 = - p1[1] / (p2[1] - p1[1])
        # this gives x0
        x0 = (1 - t0) * p1[0] + t0 * p2[0]
        # this gives p3 on the x-axis as
        p3 = [x0, 0.0]
        # If y1 > 0
        if p1[1] > 0 and p2[1] < 0:
            # Neumann for p1
            F[ek[0]] += lineintegral(p1, p3, 4, g)[0]

            # Dirichlet for p2
            A[ek[1], ek[1]] = 1 / epsilon
            F[ek[1]] = 0
        elif p1[1] < 0 and p2[1] > 0:
            # Dirichlet for p1
            A[ek[0], ek[0]] = 1 / epsilon
            F[ek[0]] = 0
            # Neumann for p2
            F[ek[1]] += lineintegral(p3, p2, 4, g)[1]
        else:
            # This should not be a possible case, but if it happens, just use Dirichlet
            A[np.ix_(ek, ek)] = 1 / epsilon
            F[ek] = 0

    return spsolve(A.tocsr(), F)


def Dirichlet2D(N, f, u_exact, epsilon=1e-6, DoSingularityCheck=True, save=False):
    print("-" * 60)
    BC_type = 'Dirichlet'
    print(BC_type + " with N = " + str(N))
    # get numerical solution
    U_dir = Dirichlet_Poisson2D(N, f, epsilon, DoSingularityCheck)
    # Create plot
    contourplot(N, U_dir, BC_type, u_exact, save=save)


def Mixed2D(N, f, g, u_exact, epsilon=1e-6, DoSingularityCheck=True, save=False):
    print("-" * 60)
    BC_type = "Mixed"
    print(BC_type + " with N = " + str(N))
    # get numerical solution
    U_mix = Mixed_Poisson2D(N, f, g, epsilon, DoSingularityCheck)
    # Create plot
    contourplot(N, U_mix, BC_type, u_exact, save=save)



if __name__ == "__main__":
    # source function
    f = lambda x, y: 16 * np.pi * np.pi * (x * x + y * y) * np.sin(2 * np.pi * (x * x + y * y)) \
                          - 8 * np.pi * np.cos(2 * np.pi * (x * x + y * y))
    # Neumann B.C. function on x^2 + y^2 = r^2 = 1
    # 4 * np.pi  * np.sqrt(x * x + y * y) * np.cos(2 * np.pi * (x * x + y * y)) = 4 * np.pi on x^2 + y^2 = r^2 = 1
    g = lambda x, y: 4 * np.pi
    # exact solution
    u_exact = lambda x, y:  np.sin(2 * np.pi * (x * x + y * y))

    # save the plots?
    save = False
    # Do the Singularity check?
    DoSingularityCheck = True
    # chose the N
    N_list = [100, 500, 1000]

    for N in N_list:
        Dirichlet2D(N, f, u_exact, epsilon=1e-6, DoSingularityCheck=DoSingularityCheck, save=save)
        Mixed2D(N, f, g, u_exact, epsilon=1e-6, DoSingularityCheck=DoSingularityCheck, save=save)







