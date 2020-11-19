# -*- coding: utf-8 -*-
"""
Created on 16.11.2020

@author: Olav Milian
"""
import numpy as np
import matplotlib.pyplot as plt
from getplate import getPlate
from Heat2D import ThetaMethod_Heat2D


import sympy as sym

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(newparams)


def contourplot_Heat2D(N, u_hdict, save_name, save=False):
    """
    Function to make contourplots of the three first entry's in u_hdict

    Parameters
    ----------
    N : int
        Number of nodes in the mesh.
    numerical_solution : numpy array (list)
        The numerical solution to the problem solved with B.C type.
    BC_type : str
        A string containing the name of the B.C. type.
    u_exact_func : function pointer
        Pointer to the function for the exact solution.
    save : bool, optional
        Will the plot be saved in the plot folder. The default is False.
        Note: the plot folder must exist!

    Returns
    -------
    None.

    """
    # get the nodes, elements and edge lines
    p, tri, edge = getPlate(N+1)

    u_h0, t0 = u_hdict[0]
    u_h1, t1 = u_hdict[1]
    u_h2, t2 = u_hdict[2]

    # x and y coordinates
    x = p[:, 0]
    y = p[:, 1]

    # restrict color coding to (-1, 1),
    levels = np.linspace(-1, 1, 9)

    plt.figure(figsize=(21, 6))
    # Create plot of numerical solution
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h0, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + "{:.2f}".format(t0) + "$ for $N=" +str(N) + "$")

    # Create plot of analytical solution
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h1, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + "{:.2f}".format(t1) + "$ for $N=" +str(N) + "$")

    # Create plot of absolute difference between the two solutions
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h2, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + "{:.2f}".format(t2) + "$ for $N=" +str(N) + "$")

    # adjust
    plt.subplots_adjust(wspace=0.4)

    # save the plot?
    if save:
        plt.savefig("plot/" + save_name + ".pdf")
    plt.show()



def meshplot_v2(N_list, nCols=4, save=False):
    """
    Function to make plots of meshes for N in N_list

    Parameters
    ----------
    N_list : list
        A list containing N's to plot a mesh for.
    nCols : int, optional
        Number of columns in the final plot, meaning subplots per row. The default is 3.
    save : bool, optional
        Will the plot be saved in the plot folder. The default is False.
        Note: the plot folder must exist!

    Returns
    -------
    None.

    """
    # if N_list is just an int
    if isinstance(N_list, int):
        N_list = [N_list]

    # create a figure
    numPlots = len(N_list)
    nRows = numPlots // nCols + numPlots % nCols
    # make N_list's length to nRows*nCols by appending -1
    while len(N_list) < nRows * nCols:
        N_list.append(-1)
    # reshape N_list
    N_list = np.array(N_list).reshape((nRows, nCols))
    # create the main figure and the axs as the subplots,
    # the figsize (21, 6) is good for nRows = 1
    # but figsize (21, 7 * nRows) is better for nRows = 2, 3, ...
    if nRows == 1:
        c = 6
    else:
        c = 7 * nRows
    fig, axs = plt.subplots(nRows, nCols, figsize=(21, c))
    if nCols == 1:
        axs = np.array([axs, ])
    if nRows == 1:
        axs = np.array([axs, ])

    for i in range(nRows):
        for j in range(nCols):
            N = N_list[i, j]
            if N == -1:
                # don't show plot
                axs[i, j].set_axis_off()
            else:
                ax = axs[i, j]
                # get the nodes, elements and edge lines
                p, tri, edge = getPlate(N+1)
                # plot them with triplot
                ax.triplot(p[:, 0], p[:, 1], tri)
                # label the axes
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                # give the plot a title
                ax.set_title('Mesh for $N=' + str(N) + '$')
    # adjust
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # save the plot?
    if save:
        plt.savefig("plot\Mesh_for_Ns.pdf")
    plt.show()


def plotError(N_list, error_dict, time_stamps, save=False):

    h_vec = np.array([1 / N for N in N_list])
    plt.figure("error", figsize=(14, 7))
    for key in error_dict:  # key is a int
        error = error_dict[key]
        t = np.round(time_stamps[key], 2)
        plt.loglog(h_vec, error, 'o-', label="$rel.error(t=" + str(t) + ")$")
        p = np.polyfit(np.log(h_vec[1:]), np.log(error[1:]), deg=1)
        print(p)
        # err2 = np.exp(p[0] * np.log(h_vec) + p[1])
        # plt.loglog(h_vec, err2)
    plt.xlabel("Element size, $h$")
    plt.ylabel("log relative error")

    plt.grid()
    plt.title("Element size v. relative error")
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.11), ncol=3)
    if save:
        plt.savefig("plot/Relative_error.pdf")
    plt.show()

def plottime(N_list, time_vec1, time_vec2, save=False):
    h_vec = np.array([1 / N for N in N_list])
    plt.figure("time", figsize=(14, 7))
    plt.plot(h_vec, time_vec2, 'o-', label="Time to find error estimate")
    plt.plot(h_vec, time_vec1, 'o-', label="Time to solve problem")

    plt.xlabel("Element size, $h$")
    plt.ylabel("Time, $t$")

    plt.grid()
    plt.title("Element size v. Time")
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.11), ncol=3)
    if save:
        plt.savefig("plot/Time.pdf")
    plt.show()


if __name__ == "__main__":
    f = lambda x, y, t, beta=5: np.exp(- beta * (x*x + y*y))

    uD = lambda x, y, t: np.zeros_like(x)
    
    duDdt = lambda x, y, t: np.zeros_like(x)
    
    u0 = lambda x, y: np.zeros_like(x)
    
    N = 16
    Nt = 34
    alpha = 9.62e-5
    beta = 3
    # save the plot as pdf?
    save = False

    save_name = "FE000"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0, T=1, f_indep_t=True)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)
    save_name = "ITrap000"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=1, f_indep_t=True)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)
    save_name = "BE000"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=1, T=1, f_indep_t=True)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)

    save_name = "ITrap000b10"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, 7, f, uD, duDdt, u0, theta=0.5, T=1, f_indep_t=True)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)

    f = lambda x, y, t, beta=5: np.exp(- beta * (x * x + y * y))

    uD = lambda x, y, t: y / 2 + 1/ 2

    duDdt = lambda x, y, t: np.zeros_like(x)

    u0 = lambda x, y: y / 2 + 1/ 2
    save_name = "ITrapy0y"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=1, f_indep_t=True)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)

    f = lambda x, y, t, beta, a=1: np.exp(- beta * ((x - a * np.sin(t)) ** 2 + y * y))

    uD = lambda x, y, t: np.zeros_like(x)

    duDdt = lambda x, y, t: np.zeros_like(x)

    u0 = lambda x, y: np.zeros_like(x)
    save_name = "ITrap000a1"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=2*np.pi, f_indep_t=False)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)

    f = lambda x, y, t, beta, a=5: np.exp(- beta * ((x - a * np.sin(t)) ** 2 + y * y))

    uD = lambda x, y, t: np.zeros_like(x)

    duDdt = lambda x, y, t: np.zeros_like(x)

    u0 = lambda x, y: np.zeros_like(x)
    save_name = "ITrap000a5"
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=2*np.pi, f_indep_t=False)
    contourplot_Heat2D(N, u_hdict, save_name, save=save)

    # list of N to plot for
    N_list = [1, 2, 4, 8]
    # make a meshplot
    meshplot_v2(N_list, save=False)


