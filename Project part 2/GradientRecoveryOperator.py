# -*- coding: utf-8 -*-
"""
Created on 18.11.2020

@author: Olav Milian
"""
import numpy as np
from timeit import default_timer as timer
from getplate import getPlate
from Gauss_quadrature import quadrature2D
from Heat2D import ThetaMethod_Heat2D
from plotingfunctions import plotError, plottime


def get_nodal_patch(node, tri):
    nodal_patch = []
    for nk in tri:
        if node in nk:
            nodal_patch.append(nk)
    nodal_patch = np.asarray(nodal_patch)
    return nodal_patch


def centroid(nk, p):
    ck = (p[nk[0], :] + p[nk[1], :] + p[nk[2], :]) / 3
    return ck

# p_vector in min problem
p_vec = lambda x, y: np.array([1, x, y])

# matrix Mk in min problem
Mk = lambda x, y: np.outer(p_vec(x, y), p_vec(x, y))


def get_alpha(u_h0, u_h1, u_h2, nk, p, tri):

    alpha_dict = {}

    for i in range(3):
        node = nk[i]
        nodal_patch = get_nodal_patch(node, tri)
        # matrix M for min problem
        M = np.zeros((3, 3))
        # vector bx and by for min problem, rows: t0, t1, t1
        bx = np.zeros((3, 3))
        by = np.zeros((3, 3))
        for nkbar in nodal_patch:
            # the centroid
            rk = centroid(nkbar, p)
            M += Mk(*rk)
            # compute the gradient of u_h the element
            # note the gradient is constant on the element by the choice of basis
            # calculate basis functions, gradient of jacobian etc.
            # row_k: [1, x_k, y_k]
            Bk = np.asarray([p_vec(*p[nkbar[0]]), p_vec(*p[nkbar[1]]), p_vec(*p[nkbar[2]])])
            Ck = np.linalg.inv(Bk)  # here faster than solving Mk @ Ck = I_3
            # x - comp of grad phi, is constant
            Ckx = Ck[1, :]
            # y - comp of grad phi, is constant
            Cky = Ck[2, :]
            # the x-component  on the element
            u_h0_x = np.sum(Ckx @ u_h0[nkbar])
            u_h1_x = np.sum(Ckx @ u_h1[nkbar])
            u_h2_x = np.sum(Ckx @ u_h2[nkbar])
            # the y-component  on the element
            u_h0_y = np.sum(Cky @ u_h0[nkbar])
            u_h1_y = np.sum(Cky @ u_h1[nkbar])
            u_h2_y = np.sum(Cky @ u_h2[nkbar])
            # build bx
            bx[:, 0] += p_vec(*rk) * u_h0_x
            bx[:, 1] += p_vec(*rk) * u_h1_x
            bx[:, 2] += p_vec(*rk) * u_h2_x
            # build by
            by[:, 0] += p_vec(*rk) * u_h0_y
            by[:, 1] += p_vec(*rk) * u_h1_y
            by[:, 2] += p_vec(*rk) * u_h2_y
        # Now solve to get alpha_x and alpha_y for node i
        try:
            alpha_x = np.linalg.solve(M, bx)
        except np.linalg.LinAlgError:
            alpha_x = np.linalg.lstsq(M, bx, rcond=None)[0]
        try:
            alpha_y = np.linalg.solve(M, by)
        except np.linalg.LinAlgError:
            alpha_y = np.linalg.lstsq(M, by, rcond=None)[0]
        alpha_dict[i+1] = [alpha_x, alpha_y]
    return alpha_dict


def Error_Estimate_G(N, u_hdict):
    # Function find the Error estimate for the three first entry's in u_hdict using the Gradient Recovery Operator

    u_h0, t0 = u_hdict[0]
    u_h1, t1 = u_hdict[1]
    u_h2, t2 = u_hdict[2]

    p, tri, edge = getPlate(N+1)
    #   p		Nodal points, (x,y)-coordinates for point i given in row i.
    #   tri   	Elements. Index to the three corners of element i given in row i.
    #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.

    # initialize error estimate
    eta2_vec = np.zeros(3)

    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        p1 = p[nk[0], :]
        p2 = p[nk[1], :]
        p3 = p[nk[2], :]
        # array for grad u_h,
        # columns ph1, phi2, phi3
        # rows: t0, t1, t1
        gradu_hx = np.zeros((3, 3))
        gradu_hy = np.zeros((3, 3))
        # array for g_x and g_y in each point, rows: t0, t1, t1
        g_x = np.zeros((3, 3))
        g_y = np.zeros((3, 3))
        alpha_dict = get_alpha(u_h0, u_h1, u_h2, nk, p, tri)
        alpha1 = alpha_dict[1]
        alpha2 = alpha_dict[2]
        alpha3 = alpha_dict[3]

        # calculate basis functions.
        # compute the gradient of u_h the element
        # note the gradient is constant on the element by the choice of basis
        # row_k: [1, x_k, y_k]
        Bk = np.asarray([p_vec(*p[nk[0]]), p_vec(*p[nk[1]]), p_vec(*p[nk[2]])])
        Ck = np.linalg.inv(Bk)  # here faster than solving Mk @ Ck = I_3
        # x - comp of grad phi, is constant
        Ckx = Ck[1, :]
        # y - comp of grad phi, is constant
        Cky = Ck[2, :]
        # the x-component  on the element
        gradu_hx[:, 0] = Ckx * u_h0[nk]
        gradu_hx[:, 1] = Ckx * u_h1[nk]
        gradu_hx[:, 2] = Ckx * u_h2[nk]
        # the y-component  on the element
        gradu_hy[:, 0] = Cky * u_h0[nk]
        gradu_hy[:, 1] = Cky * u_h1[nk]
        gradu_hy[:, 2] = Cky * u_h2[nk]
        # the basis functions
        # phi_1 = [1, x, y] @ Ck[:,0]
        # phi_2 = [1, x, y] @ Ck[:,1]
        # phi_3 = [1, x, y] @ Ck[:,2]
        phi = lambda x, y: [1, x, y] @ Ck

        # The function to integrate
        # the x-comp
        Fx = lambda x, y, i: p_vec(*p1) @ alpha1[0][:, i] * phi(x, y)[0] - gradu_hx[0, i] \
                              + p_vec(*p2) @ alpha2[0][:, i] * phi(x, y)[1] - gradu_hx[1, i]\
                              + p_vec(*p3) @ alpha3[0][:, i] * phi(x, y)[2] - gradu_hx[2, i]
        # the y-comp
        Fy = lambda x, y, i: p_vec(*p1) @ alpha1[1][:, i] * phi(x, y)[0] - gradu_hy[0, i] \
                              + p_vec(*p2) @ alpha2[1][:, i] * phi(x, y)[1] - gradu_hy[1, i] \
                              + p_vec(*p3) @ alpha3[1][:, i] * phi(x, y)[2] - gradu_hy[2, i]

        # the function of i
        Fi = lambda x, y, i: Fx(x, y, i) ** 2 + Fy(x, y, i) ** 2

        # now do the integration
        for i in range(3):
            # the function
            F = lambda x, y: Fi(x, y, i)
            eta2_vec[i] += quadrature2D(p1, p2, p3, 4, F)

    return eta2_vec


def get_error_estimate(f, uD, duDdt, u0, N_list=None, Nt=100, beta=5, alpha=9.62e-5, theta=0.5, T=1, Rg_indep_t=True, f_indep_t=True):
    if N_list is None:
        N_list = [1, 2, 4, 8, 32]
        # these values give h=1/2^i for i=0,1,2,3 and 5, see findh.py

    m = len(N_list)
    error_dict = {0: np.zeros(m), 1: np.zeros(m), 2: np.zeros(m)}  # there are 3 timestamps in u_hdict
    time_vec1 = np.zeros(m)
    time_vec2 = np.zeros(m)
    for i in range(m):
        N = N_list[i]
        start_t = timer()
        u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=theta, T=T, Rg_indep_t=Rg_indep_t, f_indep_t=f_indep_t)
        end_t = timer()
        time_vec1[i] = end_t - start_t
        print("N =", N, "took", end_t-start_t, "seconds")
        start_t = timer()
        eta2_vec = Error_Estimate_G(N, u_hdict)
        end_t = timer()
        time_vec2[i] = end_t - start_t
        print("Found error estimate for N =", N, "in", end_t-start_t, "seconds or", (end_t -start_t) / 60, "minutes")
        error_dict[0][i] = eta2_vec[0]
        error_dict[1][i] = eta2_vec[1]
        error_dict[2][i] = eta2_vec[2]

        if i == 0:
            # get the timestamps
            time_stamps = np.array([u_hdict[0][1], u_hdict[1][1], u_hdict[2][1]])

    for i in range(3):
        # the relative error
        # note eta_i / eta_5 = eta_i^2 / eta_5^2, so we can work with eta^2
        error_dict[i] = np.sqrt(error_dict[i])


    return N_list, error_dict, time_vec1, time_vec2, time_stamps

if __name__ == "__main__":

    f = lambda x, y, t, beta: np.exp(- beta * (x * x + y * y))

    uD = lambda x, y, t: np.zeros_like(x)

    duDdt = lambda x, y, t: np.zeros_like(x)

    u0 = lambda x, y: np.zeros_like(x)

    # save the plot as pdf?
    save = False
    N_list, error_dict, time_vec1, time_vec2, time_stamps = get_error_estimate(f, uD, duDdt, u0, N_list=[1, 2, 4, 8])

    print(error_dict)
    plotError(N_list, error_dict, time_stamps, save=save)

    plottime(N_list, time_vec1, time_vec2, save=save)
