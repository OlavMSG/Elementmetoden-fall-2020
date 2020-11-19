# -*- coding: utf-8 -*-
"""
Created on 19.11.2020

@author: Olav Milian
"""
import numpy as np
from timeit import default_timer as timer
from Heat2D import ThetaMethod_Heat2D
from plotingfunctions import plotError, plottime
from getplate import getPlate

def check_N(N, Nbar):
    while Nbar != N:
        if Nbar % 2 != 0:
            return False
        Nbar /= 2
    return True


def interpolation(ec, M):
    # Taken from Olav Gran's Numerical Linear Algebra Project, TMA4205
    # Using linear interpolation.
    # function to do the prolongation/interploation of the error
    # ec - the error to interpolate
    # M - the size of the rhs matrix above is (M+1)x(M+1)

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
    ef[f_ixy] = ec[c_ixy]

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
    ef[f_ixp_y] = 0.5 * (ec[c_ixy_xm1] + ec[c_ixp_y])
    ef[f_ix_yp] = 0.5 * (ec[c_ixy_ym1] + ec[c_ix_yp])
    ef[f_ixp_yp] = 0.25 * (ec[c_ixy_xym1] + ec[c_ixp_y_ym1] + ec[c_ix_yp_xm1] + ec[c_ixp_yp])
    return ef


def Estimate_Error(u_hNdict, N_list):

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

    m = len(N_list)
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

        k = np.int(np.log2(Nbar // N))
        u_h0int = u_h0.copy()
        u_h1int = u_h1.copy()
        u_h2int = u_h2.copy()
        M = N
        for j in range(k):
            # Use linear interpolation to get to the grid 2*M
            # Do this until 2*M = Nbar
            # Note this is equivalent to interpolating directly up to Nbar.
            u_h0int = interpolation(u_h0int, M)
            u_h1int = interpolation(u_h1int, M)
            u_h2int = interpolation(u_h2int, M)
            M *= 2

        err0 = u_h0bar - u_h0int
        err1 = u_h1bar - u_h1int
        err2 = u_h2bar - u_h2int

        error_dict[0][i] = np.linalg.norm(err0, ord="fro") / np.linalg.norm(u_h0bar, ord="fro")
        error_dict[1][i] = np.linalg.norm(err1, ord="fro") / np.linalg.norm(u_h1bar, ord="fro")
        error_dict[2][i] = np.linalg.norm(err2, ord="fro") / np.linalg.norm(u_h2bar, ord="fro")

    return error_dict


def get_error_estimate(f, uD, duDdt, u0, N_list=None, Nt=100, beta=5, alpha=9.62e-5, theta=0.5, T=1, Rg_indep_t=True, f_indep_t=True):
    global time_stamps
    if N_list is None:
        N_list = [1, 2, 4, 8, 32]
        # these values give h=1/2^i for i=0,1,2,3 and 5, see findh.py

    m = len(N_list)
    time_vec1 = np.zeros(m)
    u_hNdict = {}
    for i in range(m):
        N = N_list[i]
        start_t = timer()
        u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=theta, T=T, Rg_indep_t=Rg_indep_t, f_indep_t=f_indep_t)
        end_t = timer()
        u_hNdict[N] = u_hdict
        time_vec1[i] = end_t - start_t
        print("N =", N, "took", end_t-start_t, "seconds")

        if i == 0:
            # get the timestamps
            time_stamps = np.array([u_hdict[0][1], u_hdict[1][1], u_hdict[2][1]])

    error_dict = Estimate_Error(u_hNdict, N_list)

    return N_list, error_dict, time_vec1, time_stamps

if __name__ == "__main__":

    f = lambda x, y, t, beta: np.exp(- beta * (x * x + y * y))

    uD = lambda x, y, t: np.zeros_like(x)

    duDdt = lambda x, y, t: np.zeros_like(x)

    u0 = lambda x, y: np.zeros_like(x)

    # save the plot as pdf?
    save = False
    N_list, error_dict, time_vec1, time_stamps = get_error_estimate(f, uD, duDdt, u0, N_list=[1, 2, 4, 8])

    print(error_dict)
    plotError(N_list[:-1], error_dict, time_stamps, "Rel", "relative Error", save=save)

    plottime(N_list, "Rel", time_vec1, save=save)
