# -*- coding: utf-8 -*-
"""
Created on 19.11.2020

@author: Olav Milian
"""
import numpy as np
from timeit import default_timer as timer
from Heat2D import ThetaMethod_Heat2D


def check_N(N, Nbar):
    """
    Function to check if Nbar can be devided by only 2 multiple times to get N

    Parameters
    ----------
    N : int
        Number we want to now is is factor of Nbar in form Nbar = 2^i*N.
    Nbar : int
        Number we want to decide by 2 multiple times.

    Returns
    -------
    bool
        Cab Nbar we written as Nbar = 2^i*N.

    """
    while Nbar != N:
        if Nbar % 2 != 0:
            return False
        Nbar /= 2
    return True


def interpolation(ec, M):
    """
    Function to do linear interpolation from coarse grid M to fine grid N = 2 * M
    Note: Taken from Olav Milian Schmitt Gran's Numerical Linear Algebra Project, TMA4205

    Parameters
    ----------
    ec : numpy.array
        matrix to be linear interpolated, size (M+1)x(M+1).
    M : int
        size of the matrix.

    Returns
    -------
    ef : numpy.array
        interpolation of ec, matrix of size (N+1)x(N+1).

    """
    # N, assumed that N is even
    # Note if N was odd this does not work!
    N = 2 * M
    ef = np.zeros((N + 1, N + 1))
    
    # Coarse and fine index
    c_index = np.arange(M + 1)
    f_2index = 2 * c_index
    
    # Indecies
    c_ixy = np.ix_(c_index, c_index)
    f_ixy = np.ix_(f_2index, f_2index)

    c_ixy_xm1 = np.ix_(c_index[:-1], c_index)
    c_ixp_y = np.ix_(c_index[:-1] + 1, c_index)
    f_ixp_y = np.ix_(f_2index[:-1] + 1, f_2index)

    c_ixy_ym1 = np.ix_(c_index, c_index[:-1])
    c_ix_yp = np.ix_(c_index, c_index[:-1] + 1)
    f_ix_yp = np.ix_(f_2index, f_2index[:-1] + 1)
    
    c_ixy_xym1 = np.ix_(c_index[:-1], c_index[:-1])
    c_ixp_y_ym1 = np.ix_(c_index[:-1] + 1, c_index[:-1])
    c_ix_yp_xm1 = np.ix_(c_index[:-1], c_index[:-1] + 1)
    c_ixp_yp = np.ix_(c_index[:-1] + 1, c_index[:-1] + 1)
    f_ixp_yp = np.ix_(f_2index[:-1] + 1, f_2index[:-1] + 1)

    # the interpolation
    ef[f_ixy] = ec[c_ixy] # known nodes
    ef[f_ixp_y] = 0.5 * (ec[c_ixy_xm1] + ec[c_ixp_y])
    ef[f_ix_yp] = 0.5 * (ec[c_ixy_ym1] + ec[c_ix_yp])
    ef[f_ixp_yp] = 0.25 * (ec[c_ixy_xym1] + ec[c_ixp_y_ym1] + ec[c_ix_yp_xm1] + ec[c_ixp_yp])
    return ef


def Estimate_Error(u_hNdict, N_list):
    """
    Function to make a error estimate

    Parameters
    ----------
    u_hNdict : dictionary
        dictionary containing multiple dictionary of u_hdict.
        u_hdict : dictionary
        dictionary of three (u_h, t_i)-values, where t_i is the time stamp for when u_h is.
     N_list : list
        A list containing N's to plot a mesh for.
        N : int
        Number of nodal edges on the x-axis.

    Raises
    ------
    ValueError
        If the N-s in N_list must be multiples of 2 of the first element in N_list..

    Returns
    -------
    error_dict : dictionary
        dictionary of the relative errors for three different time stamps.

    """

    Nbar = N_list[-1]
    u_hdictbar = u_hNdict[Nbar]

    if not check_N(N_list[0], Nbar):
        raise ValueError("The N-s in N_list must be multiples of 2 of the first element in N_list.")

    # Note that the 2-norm  coincides with the Frobenious norm in matrix form.
    # Also the nodes follow the natural order
    # meaning index = i + (N+1) * j in the u_h vector
    # so reshape to matrix for easy restriction of largest grid to smaller ones.

    # get u_h and reshape to matrix
    u_h0bar = u_hdictbar[0][0].reshape((Nbar+1, Nbar+1))
    u_h1bar = u_hdictbar[1][0].reshape((Nbar+1, Nbar+1))
    u_h2bar = u_hdictbar[2][0].reshape((Nbar+1, Nbar+1))

    # number of N-s
    m = len(N_list)
    # initilize error_dict
    error_dict = {0: np.zeros(m-1), 1: np.zeros(m-1), 2: np.zeros(m-1)}  # there are 3 timestamps in u_hdict

    for i in range(m - 1):
        N = N_list[i]
        if not check_N(N, Nbar):
            raise ValueError("The N-s in N_list must be multiples of 2 of the first element in N_list.")
        u_hdict = u_hNdict[N]
        # get u_h and reshape to matrix
        u_h0 = u_hdict[0][0].reshape((N+1, N+1))
        u_h1 = u_hdict[1][0].reshape((N+1, N+1))
        u_h2 = u_hdict[2][0].reshape((N+1, N+1))

        # find how many times we must multiply N by 2 to get Nbar
        k = np.int(np.log2(Nbar // N))

        # initial setup
        M = N
        for j in range(k):
            # Use linear interpolation to get to the grid 2*M
            # Do this until 2*M = Nbar
            # Note this is equivalent to interpolating directly up to Nbar.
            u_h0 = interpolation(u_h0, M)
            u_h1 = interpolation(u_h1, M)
            u_h2 = interpolation(u_h2, M)
            M *= 2
        
        # the error
        err0 = u_h0bar - u_h0
        err1 = u_h1bar - u_h1
        err2 = u_h2bar - u_h2
        # save the relative error
        error_dict[0][i] = np.linalg.norm(err0, ord="fro") / np.linalg.norm(u_h0bar, ord="fro")
        error_dict[1][i] = np.linalg.norm(err1, ord="fro") / np.linalg.norm(u_h1bar, ord="fro")
        error_dict[2][i] = np.linalg.norm(err2, ord="fro") / np.linalg.norm(u_h2bar, ord="fro")

    return error_dict


def get_error_estimate(f, uD, duDdt, u0, Nt, N_list=None, beta=5, alpha=9.62e-5, theta=0.5, T=2*np.pi,
                       Rg_indep_t=True, f_indep_t=False):
    """
    Function to get the error estimate using the theta method.
    Note: function is faster given that Rg_indep_t=True and f_indep_t=True, 
    meaning we assume that the boundary function and the source function are independent of t.

    Parameters
    ----------
    f : function pointer
        The source function.
    uD : function pointer
        Value of u_h on the boundary.
    duDdt : function pointer
        Derivative of u_h on the boundary, derivative of uD.
    u0 : function pointer
        initial function for t=0.
    Nt : int
        number of time steps
    N_list : list, optional
        list of N-s. The default is None.
    beta : float, optional
        parameter beta of the source function. The default is 5.
    alpha : float, optional
        parameter alpha of the equation. The default is 9.62e-5.
    heta : float, optional
        parameter for the time integration method.
        0: Forward Euler
        0.5: Implicit Traps
        1: Backward Euler
        The default is 0.5.
    T : float, optional
        The end time of the interval [0, T]. The default is 2*np.pi.
    Rg_indep_t : bool, optional
        Is the boundary function independent of t. The default is True.
    f_indep_t : bool, optional
        Is the source function independent of t. The default is False.

    Returns
    -------
    N_list : list
        list of N-s.
    error_dict : dictionary
        dictionary of relative errors.
    time_vec1 : numpy.array
        Array of times to solve the heat equation.
    time_stamps : numpy.array
        Array of the time stamps.

    """
    if N_list is None:
        N_list = [1, 2, 4, 8, 32]
    m = len(N_list)
    time_vec1 = np.zeros(m)
    u_hNdict = {}
    for i in range(m):
        N = N_list[i]
        start_t = timer()
        u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=theta, T=T,
                                     Rg_indep_t=Rg_indep_t, f_indep_t=f_indep_t)
        end_t = timer()
        u_hNdict[N] = u_hdict
        time_vec1[i] = end_t - start_t
        print("N =", N, "took", end_t-start_t, "seconds")

        if i == 0:
            # get the timestamps
            time_stamps = np.array([u_hdict[0][1], u_hdict[1][1], u_hdict[2][1]])

    error_dict = Estimate_Error(u_hNdict, N_list)

    return N_list, error_dict, time_vec1, time_stamps

