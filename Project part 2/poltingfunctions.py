# -*- coding: utf-8 -*-
"""
Created on 16.11.2020

@author: Olav Milian
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from getdisc import GetDisc
from Heat2D import ThetaMethod_Heat2D


import sympy as sym

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(newparams)


def contourplot_Heat2D(N, u_hdict, save=False):
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
    p, tri, edge = GetDisc(N)

    u_h0, t0 = u_hdict[0]
    u_h1, t1 = u_hdict[1]
    u_h2, t2 = u_hdict[2]

    # x and y coordinates
    x = p[:, 0]
    y = p[:, 1]

    # restrict color coding to (-1, 1),
    levels = np.linspace(-1, 1, 9)

    plt.figure(figsize=(21, 6))
    plt.title('$u_h(t, x,y)$, for $N=' + str(N) + "$")
    # Create plot of numerical solution
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h0, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + str(t0) + "$")

    # Create plot of analytical solution
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h1, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + str(t1) + "$")

    # Create plot of absolute difference between the two solutions
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h2, levels=levels, extend='both')
    plt.colorbar()

    plt.title("$u_h(t, x,y)$ at $t=" + str(t2) + "$")

    # adjust
    plt.subplots_adjust(wspace=0.4)

    # save the plot?
    if save:
        save_name = "plot\Heat_N=" + str(N)
        plt.savefig(save_name + ".pdf")
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
            print(N)
            if N == -1:
                # don't show plot
                axs[i, j].set_axis_off()
            else:
                ax = axs[i, j]
                # get the nodes, elements and edge lines
                p, tri, edge = GetDisc(N)
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
        plt.savefig("plot\Mesh_for_N.pdf")
    plt.show()


if __name__ == "__main__":
    """f = lambda x, y, t, beta: np.exp(- beta * (x*x + y*y))

    uD = lambda x, y, t: np.zeros_like(x)
    
    duDdt = lambda x, y, t: np.zeros_like(x)
    
    u0 = lambda x, y: np.zeros_like(x)
    
    N = 500
    Nt = 250
    alpha = 9.62e-5
    beta = 1
    u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uD, duDdt, u0, theta=0.5, T=10, f_indep_t=True)
    
    contourplot_Heat2D(N ,u_hdict)"""
    # save the plot as pdf?
    save = False
    # list of N to plot for
    N_list = [7, 23, 78, 274]
    print(N_list)
    # make a meshplot
    meshplot_v2(N_list, save=save)


