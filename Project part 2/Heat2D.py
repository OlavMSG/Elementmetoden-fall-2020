# -*- coding: utf-8 -*-
"""
Created on 09.11.2020

@author: Olav Gran
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from getdisc import GetDisc
from Gauss_quadrature import quadrature2D



def get_in_and_edge_index(tri, edge):
    edge_index = np.unique(edge)
    un_index = np.unique(tri)
    in_index = np.setdiff1d(un_index, edge_index)
    return in_index, edge_index


def build_Mk_Ak_F0_local(p1, p2, p3, f, alpha, beta, in_point, len_argk):
    # calculate basis functions,.
    # row_k: [1, x_k, y_k]
    Bk = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, p3[0], p3[1]]])
    Ck = np.linalg.inv(Bk)  # here faster than solving Mk @ Ck = I_3
    # phi_1 = [1, x, y] @ Ck[:,0]
    # phi_2 = [1, x, y] @ Ck[:,1]
    # phi_3 = [1, x, y] @ Ck[:,2]
    phi = lambda x, y: [1, x, y] @ Ck

    # Calculate the local stifness matrix Ak
    # Ak_{a,b} = Area_k * (Cxk_a * Cx_b + Cyk_a * Cyk_b)
    Area_k_func = lambda x, y: 1
    Area_k = quadrature2D(p1, p2, p3, 1, Area_k_func)  # 1-point rule is sufficient here
    # Ck[1:3, :].T = [[Cxk_1, Cyk_1], [Cxk_2, Cy_2], [Cxk_3, Cyk_3]]
    # Ck[1:3, :] = [[Cxk_1, Cxk_2, Cxk_3], [Cyk_1, Cyk_2, Cyk_3]]
    # (Ck[1:3, :].T @ Ck[1:3, :])_{a, b} = Cxk_a * Cxk_b + Cyk_a * Cyk_b, (a,b) in {1, 2, 3}
    Ak = Area_k * alpha * Ck[1:3, :].T @ Ck[1:3, :]

    # Calculate the mass matrix Mk and F_local
    Mk = np.zeros((3, 3))
    F0_local = np.zeros(len_argk)
    k = 0
    for i in range(3):
        if in_point[i] != -1:
            f_phi_i = lambda x, y: f(x, y, 0, beta) * phi(x, y)[i]
            F0_local[k] = quadrature2D(p1, p2, p3, 4, f_phi_i)
            k += 1
        for j in range(3):
            g_ij = lambda x, y: phi(x, y)[i] * phi(x, y)[j]
            Mk[i, j] = quadrature2D(p1, p2, p3, 4, g_ij)

    return Ak, Mk, F0_local


def base_Heat2D(N, p, tri, N_in, in_index, edge_index, f, alpha, beta):
    # Full-Stiffness matrix
    Abar = sparse.lil_matrix((N, N))
    # Full-Mass matrix
    Mbar = sparse.lil_matrix((N, N))
    # Interior-load vector
    F0 = np.zeros(N_in)

    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        p1 = p[nk[0], :]
        p2 = p[nk[1], :]
        p3 = p[nk[2], :]
        # check if pi is a inner point
        # -1 means it is an edge point
        in_point = np.arange(3)
        for i in range(3):
            if nk[i] in edge_index:
                in_point[i] = -1
        # index for nk in interior
        argk = np.nonzero(in_point != -1)[0]
        # Add the Ak and MK to the global stiffness matrix A and global mass matrix M
        indexk = np.ix_(nk, nk)
        Ak, Mk, F0_local = build_Mk_Ak_F0_local(p1, p2, p3, f, alpha, beta, in_point, len(argk))
        Abar[indexk] += Ak
        Mbar[indexk] += Mk
        # Add the local load vector Fk to the global load vector F
        F0[nk[argk]] += F0_local

    # interior and edge index
    inxy_index = np.ix_(in_index, in_index)
    inx_edgey_index = np.ix_(in_index, edge_index)
    # Stiffness matrix A and Abar contribution too F from boundary condition
    A = Abar[inxy_index]
    Ba = Abar[inx_edgey_index]

    # Mass matrix M and Mbar contribution too F from boundary condition
    M = Mbar[inxy_index]
    Bm = Mbar[inx_edgey_index]

    return A, M, F0, Ba, Bm


def build_Ft(N_in, p, tri, edge_index, f, beta, t):
    # load vector
    Ft = np.zeros(N_in)

    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        p1 = p[nk[0], :]
        p2 = p[nk[1], :]
        p3 = p[nk[2], :]
        # check if pi is a inner point
        # -1 means it is an edge point
        in_point = np.arange(3)
        for i in range(3):
            if nk[i] in edge_index:
                in_point[i] = -1
        # index for nk in interior
        argk = np.nonzero(in_point != -1)[0]
        # calculate basis functions.
        # row_k: [1, x_k, y_k]
        Bk = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, p3[0], p3[1]]])
        Ck = np.linalg.inv(Bk)  # here faster than solving Mk @ Ck = I_3
        # phi_1 = [1, x, y] @ Ck[:,0]
        # phi_2 = [1, x, y] @ Ck[:,1]
        # phi_3 = [1, x, y] @ Ck[:,2]
        phi = lambda x, y: [1, x, y] @ Ck

        Ft_local = np.zeros(len(argk))
        k = 0
        for i in range(3):
            if in_point[i] != -1:
                f_phi_i = lambda x, y: f(x, y, t, beta) * phi(x, y)[i]
                Ft_local[k] = quadrature2D(p1, p2, p3, 4, f_phi_i)
                k += 1

        # Add the local load vector Fk to the global load vector F
        Ft[nk[argk]] += Ft_local

    return Ft

def get_u_h(N, u_h1, Rg, in_index, edge_index):
    u_h = np.zeros(N)
    u_h[in_index] = u_h1
    u_h[edge_index] = Rg
    return u_h


def ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=1, Rg_indep_t=True, f_indep_t=True):
    p, tri, edge = GetDisc(N)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.

    # get lists of unique interior and edge point indices
    in_index, edge_index = get_in_and_edge_index(tri, edge)
    # number of interior nodes
    N_in = in_index.shape[0]
    # x and y on the edge
    xvec_edge = p[edge_index][:, 0]
    yvec_edge = p[edge_index][:, 1]
    # x and y on the interior
    xvec_in = p[in_index][:, 0]
    yvec_in = p[in_index][:, 0]

    # the time-step
    k = T / Nt
    ktheta = k * theta
    # thetabar
    kthetabar = k * (1 - theta)

    # get the base Stiffness Matrix, Mass matrix, F(t=0), contribution Matrices to F.
    A, M, F0, Ba, Bm = base_Heat2D(N, p, tri, N_in, in_index, edge_index, f, alpha, beta)

    # initial setup
    # u_h1(t=0), (homogenous Dirichlet)
    u_h1_current = np.zeros(N_in)
    u_h1_current[in_index] = u0(xvec_in, yvec_in)
    # The lifting function and its t derivative at t=0
    Rg_current = uD(xvec_edge, yvec_edge, t=0)
    dRgdt_current = duDdt(xvec_edge, yvec_edge, t=0)
    # The t=0 load vector with BC contributions
    F_current = F0 - Bm @ dRgdt_current - Ba @ Rg_current

    u_hdict = dict()

    # save time-picture
    # u_hdict[0] = [get_u_h(N, u_h1_current, Rg_current, in_index, edge_index), 0]
    savecount = 0

    # do iterations of theta-method
    for j in range(1, Nt+1):
        # the time
        tk = j * k
        # the left-hand side matrix
        lhs = (M + ktheta * A).tocsr()
        # The lifting function and its t derivative at t=tk
        if Rg_indep_t:
            # independent of t
            # next == current
            Rg_next = Rg_current
            dRgdt_next = dRgdt_current
        else:
            # dependent on t
            Rg_next = uD(xvec_edge, yvec_edge, t=tk)
            dRgdt_next = duDdt(xvec_edge, yvec_edge, t=tk)
        # the next load vector
        if f_indep_t:
            # independent of t
            # next == current
            F_next = F_current
        else:
            # dependent on t
            F_next = build_Ft(N_in, p, tri, edge_index, f, beta, t=tk) - Bm @ dRgdt_next - Ba @ Rg_next

        # the right-hand side vector
        rhs = (M - kthetabar * A) @ u_h1_current + ktheta * F_next + kthetabar * F_current
        # solve(lhs, rhs)
        u_h1_next = spsolve(lhs, rhs)

        # save time-picture
        if j in (Nt, int(Nt/3), int(2 * Nt / 3)):
            u_hdict[savecount] = [get_u_h(N, u_h1_next, Rg_next, in_index, edge_index), tk]
            savecount += 1

        # update
        u_h1_current = u_h1_next
        Rg_current = Rg_next
        F_current = F_next

    return u_hdict




f = lambda x, y, t, beta: np.exp(- beta * (x*x + y*y))

uD = lambda x, y, t: np.zeros_like(x)

duDdt = lambda x, y, t: np.zeros_like(x)

u0 = lambda x, y: np.zeros_like(x)

N = 15
Nt = 10
alpha = 1
beta = 1
u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5)





