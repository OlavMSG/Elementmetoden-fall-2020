# -*- coding: utf-8 -*-
"""
Created on 09.11.2020

@author: Olav Gran
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from getplate import getPlate
from Gauss_quadrature import quadrature2D



def get_in_and_edge_index(tri, edge):
    """
    Function to get the index of the inner and edge nodes

    Parameters
    ----------
    tri : numpy.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    edge : numpy.array
        Index list of all nodal points on the outer edge (r=1).

    Returns
    -------
    in_index : numpy.array
        Array of the index of the inner nodes.
    edge_index : numpy.array
        Array of the index for the edge nodes.

    """	
    # find the unique edge index
    edge_index = np.unique(edge)
    # find the unique indexes
    un_index = np.unique(tri)
    # inner index is unique index minus edge indexes
    in_index = np.setdiff1d(un_index, edge_index)
    return in_index, edge_index

def indexmap(N_in, in_index):
    """
    Function to get a map, mapping the inner indexes to 0 - N_in-1

    Parameters
    ----------
    N_in : int
        Number of inner nodes.
    in_index : numpy.array
        index of the inner nodes.

    Returns
    -------
    index_map : dictionary
        Dictionary used as a index map.

    """
    # define a dictionary for the index mapping from in_index to 0 - N_in-1
    index_map = dict(zip(in_index, np.arange(N_in)))
    return index_map

def imapfunc(indexs, index_map):
    """
    Function that given a index map maps the indexes via the index map

    Parameters
    ----------
    indexs : numpy.array
        Array of indexes.
    index_map : dictionary
        Dictionary used as a index map..

    Returns
    -------
    numpy.array/list
        Array of mapped inedxes (0 - N_in-1).

    """
    # check if indexes is empty
    if len(indexs) > 0:
        # map the indexes via the index map
        args = np.zeros_like(indexs)
        for i in range(len(indexs)):
            args[i] = index_map[indexs[i]]
        return args
    else:
        # indexes is empty
        return []


def get_u_h(N, u_h1, Rg, in_index, edge_index):
    """
    Function to build u_h from u_h1 on the inner nodes and Rg on the edge nodes

    Parameters
    ----------
    N : int
        Number of nodal edges on the x-axis.
    u_h1 : numpy.array
        The value of u_h on the inner nodes.
    Rg : numpy.array
        The value of u_h on the edge nodes.
    in_index : numpy.array
        Array of the index of the inner nodes.
    edge_index : numpy.array
        Array of the index for the edge nodes.

    Returns
    -------
    u_h : numpy.array
        The nummerical solution u_h.

    """
    N2 = (N + 1) * (N + 1)
    # initialize u_h
    u_h = np.zeros(N2)
    # put the inner node values in u_h
    u_h[in_index] = u_h1
    # put the edge node values in u_h
    u_h[edge_index] = Rg
    return u_h


def build_Mk_Ak_F0_local(p1, p2, p3, f, alpha, beta, in_point, N_inner):
    """
    Function to build the local mass-, stiffness-matrix 
    and load vector (on the inner nodes)

    Parameters
    ----------
    p1 : list
        First vertex of the triangle element.
    p2 : list
        Second vertex of the triangle element.
    p3 : list
        Third vertex of the triangle element.
    f : function pointer
        The source function.
    alpha : float
        parameter alpha of the equation.
    beta : float
        parameter beta of the source function.
    in_index : numpy.array
        Array of the index of the inner nodes.
    N_inner : int
        Number of inner nodes.

    Returns
    -------
    Ak : numpy.array
        local stiffness matrix.
    Mk : numpy.array
        local mass matrix.
    F0_local : numpy.array
        load vector on the inner nodes, maybe empty is all vertexes are edge nodes.

    """
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
    F0_local = np.zeros(N_inner)
    # index variable
    k = 0
    for i in range(3):
        if in_point[i] != -1:
            # function to integrate
            f_phi_i = lambda x, y: f(x, y, 0, beta) * phi(x, y)[i]
            F0_local[k] = quadrature2D(p1, p2, p3, 4, f_phi_i)
            k += 1
        for j in range(3):
            # function to integrate
            g_ij = lambda x, y: phi(x, y)[i] * phi(x, y)[j]
            Mk[i, j] = quadrature2D(p1, p2, p3, 4, g_ij)
    return Ak, Mk, F0_local


def base_Heat2D(N, p, tri, N_in, in_index, edge_index, f, alpha, beta):
    """
    Function to get the base of the heat equation in 2D, 
    and the load vector at t=0 for the inner nodes

    Parameters
    ----------
    N : int
        Number of nodal edges on the x-axis.
    p : numpy.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : numpy.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    N_in : int
        Number of inner nodes.
    in_index : numpy.array
        Array of the index of the inner nodes.
    edge_index : numpy.array
        Array of the index for the edge nodes.
    f : function pointer
        The source function.
    alpha : float
        parameter alpha of the equation.
    beta : float
        parameter beta of the source function

    Returns
    -------
    A : scipy.sparse.lil_matrix
        Stiffness matrix for the inner nodes.
    M : scipy.sparse.lil_matrix
        Mass matrix for the inner nodes.
    F0 : numpy.array
        Load vector on the inner nodes.
    Ba : scipy.sparse.lil_matrix
        The part of the stiffness matrix that works on the edge nodes.
    Bm : scipy.sparse.lil_matrix
        The part of the mass matrix that works on the edge nodes.

    """
    N2 = (N + 1) * (N + 1)
    # Full-Stiffness matrix
    Abar = sparse.lil_matrix((N2, N2))
    # Full-Mass matrix
    Mbar = sparse.lil_matrix((N2, N2))
    # Interior-load vector
    F0 = np.zeros(N_in)
    index_map = indexmap(N_in, in_index)
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
        args = imapfunc(nk[argk], index_map)
        F0[args] += F0_local
    # interior and edge index
    inxy_index = np.ix_(in_index, in_index)
    inx_edgey_index = np.ix_(in_index, edge_index)
    # Stiffness matrix A and Abar contribution too F from boundary condition
    A = Abar[inxy_index]
    Ba = Abar[inx_edgey_index]
    # Mass matrix M and Mbar contribution too F from boundary condition
    M = Mbar[inxy_index]
    Bm = Mbar[inx_edgey_index]
    # return
    return A, M, F0, Ba, Bm


def build_Ft(N_in, p, tri, in_index, edge_index, f, beta, t):
    """
    Function to build the load vector on the inner nodes given a time t

    Parameters
    ----------
    N_in : int
        Number of inner nodes.
    p : numpy.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : numpy.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    in_index : numpy.array
        Array of the index of the inner nodes.
    edge_index : numpy.array
        Array of the index for the edge nodes.
    f : function pointer
        The source function.
    beta : float
        parameter beta of the source function
    t : float
        the time t.

    Returns
    -------
    Ft : numpy.array
        the load vector on the inner nodes given a time t.

    """
    # load vector
    Ft = np.zeros(N_in)
    index_map = indexmap(N_in, in_index)
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
        # initialize Ft_local
        Ft_local = np.zeros(len(argk))
        k = 0
        for i in range(3):
            if in_point[i] != -1:
                f_phi_i = lambda x, y: f(x, y, t, beta) * phi(x, y)[i]
                Ft_local[k] = quadrature2D(p1, p2, p3, 4, f_phi_i)
                k += 1
        # Add the local load vector Fk to the global load vector F
        args = imapfunc(nk[argk], index_map)
        Ft[args] += Ft_local
    return Ft


def ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=1, Rg_indep_t=True, f_indep_t=False):
    """
    Function solve the Heat equartion in 2D using the theta methoid and to get u_h values for three different time stamps (1/3, 2/3, 1) * T
    Note: function is faster given that Rg_indep_t=True and f_indep_t=True, meaning we assume that the boundary function and the source function are independent of t.

    Parameters
    ----------
    N : int
        Number of nodal edges on the x-axis.
    Nt : int
        Number of time steps.
    alpha : float
        parameter alpha of the equation.
    beta : float
        parameter beta of the source function
    f : function pointer
        The source function.
    uD : function pointer
        Value of u_h on the boundary.
    duDdt : function pointer
        Derivative of u_h on the boundary, derivative of uD.
    u0 : function pointer
        initial function for t=0.
    theta : float, optional
        parameter for the time itegration method.
        0: Forward Euler
        0.5: Implicit Trapes
        1: Backward Euler
        The default is 0.5.
    T : float, optional
        The end time of the interval [0, T]. The default is 1.
    Rg_indep_t : bool, optional
        Is the boundary function independent of t. The default is True.
    f_indep_t : bool, optional
        Is the source function independent of t. The default is False.

    Returns
    -------
    u_hdict : dictionary
        dictionary of three (u_h, t_i)-values, where t_i is the time stamp for when u_h is.

    """
    p, tri, edge = getPlate(N+1)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Index list of all nodal points on the outer edge (r=1).
    # get lists of unique interior and edge point indices
    in_index, edge_index = get_in_and_edge_index(tri, edge)
    # number of interior nodes
    N_in = in_index.shape[0]
    # x and y on the edge
    xvec_edge = p[edge_index][:, 0]
    yvec_edge = p[edge_index][:, 1]
    # x and y on the interior
    xvec_in = p[in_index][:, 0]
    yvec_in = p[in_index][:, 1]
    # the time-step
    k = T / Nt
    ktheta = k * theta
    # thetabar
    kthetabar = k * (1 - theta)
    # get the base Stiffness Matrix, Mass matrix, F(t=0), contribution Matrices to F.
    A, M, F0, Ba, Bm = base_Heat2D(N, p, tri, N_in, in_index, edge_index, f, alpha, beta)
    # initial setup
    # u_h1(t=0), (homogenous Dirichlet)
    u_h1_current = u0(xvec_in, yvec_in)
    # The lifting function and its t derivative at t=0
    Rg_current = uD(xvec_edge, yvec_edge, t=0)
    dRgdt_current = duDdt(xvec_edge, yvec_edge, t=0)
    # The t=0 load vector with BC contributions
    F_current = F0 - Bm @ dRgdt_current - Ba @ Rg_current
    # dictionary to save to
    u_hdict = dict()
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
            F_next = build_Ft(N_in, p, tri, in_index, edge_index, f, beta, t=tk) - Bm @ dRgdt_next - Ba @ Rg_next
        # the right-hand side vector
        rhs = (M - kthetabar * A) @ u_h1_current + ktheta * F_next + kthetabar * F_current
        # solve(lhs, rhs)
        u_h1_next = spsolve(lhs, rhs)
        # save time-picture, we have choosen to save three.
        if j in (int(Nt/3), int(2 * Nt / 3), Nt):
            u_hdict[savecount] = [get_u_h(N, u_h1_next, Rg_next, in_index, edge_index), tk]
            savecount += 1
        # update
        u_h1_current = u_h1_next
        Rg_current = Rg_next
        F_current = F_next
    # return
    return u_hdict





