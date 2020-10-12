# -*- coding: utf-8 -*-
"""
Created on 07.10.2020

@author: Olav Gran
in collaboration with Ruben Mustad
(based in old code by Ruben)
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from getdisc import GetDisc

import sympy as sym
"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

def contourplot(N, numerical_solution, BC_type, u_exact_func, save=False):
    """
    Function to make three contorplots*: the nummerical solution, the exact solution and the absoulute error

    Parameters
    ----------
    N : int
        Number of nodes in the mesh.
    numerical_solution : numpy array (list)
        The nummerical solution to the problem solved with B.C type.
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

    # x and y coordinates
    x = p[:, 0]
    y = p[:, 1]
    u_exact = u_exact_func(x, y)
    # the absolute error
    abs_err = np.abs(u_exact - numerical_solution)

    # restrict color coding to (-1, 1),
    levels = np.linspace(-1, 1, 9)

    plt.figure(figsize=(21, 6))
    # Create plot of numerical solution
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, numerical_solution, levels=levels, extend='both')
    plt.colorbar()
    plt.title('Numerical solution, $u_h$, for\n$N=' + str(N) + '$ with ' + str(BC_type) + ' B.C')

    # Create plot of analytical solution
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_exact, levels=levels, extend='both')
    plt.colorbar()
    plt.title('Exact solution, $u$, for\n$N=' + str(N) + '$ with ' + str(BC_type) + ' B.C')

    # Create plot of absolute difference between the two solutions
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, abs_err, extend='max')
    plt.colorbar()

    plt.title('Absolute error, $|u-u_h|$, for\n$N=' + str(N) + '$ with ' + str(BC_type) + ' B.C.')

    # adjust
    plt.subplots_adjust(wspace=0.4)
    
    # save the plot?
    if save:
        save_name = "plot\Poisson_N=" + str(N) + "_with_" + BC_type + "_BC"
        plt.savefig(save_name + ".pdf")
    plt.show()


def meshplot(N_list, nCols=3, save=False):
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
    # save the plot as pdf?
    save = False
    # list of N to plot for
    N_list = [100, 500, 1000]
    # make a meshplot
    meshplot(N_list, save=save)

