# -*- coding: utf-8 -*-
"""
Created on 07.10.2020

@author: Olav Milian
"""
import numpy as np
import matplotlib.pyplot as plt
from getdisc import GetDisc

import sympy as sym
"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (21, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

def contourplot(N, numerical_solution, BC_type, u_exact_func, save=False):
    # get the nodes, elements and edge lines
    p, tri, edge = GetDisc(N)

    # get the max and min of the computed solution
    """print('The maximum value of the computed solution is: ' + str(
        max(numerical_solution)) + ' and the \n minimum value of the computed solution is: '
          + str(min(numerical_solution)) + ' for ' + str(A) + ' B.C')"""

    # x and y coordinates
    x = p[:, 0]
    y = p[:, 1]
    u_exact = u_exact_func(x, y)

    # Create plot of numerical solution
    plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, numerical_solution)
    plt.colorbar()
    plt.title('Numerical solution for N=' + str(N) + '\nwith ' + str(BC_type) + ' B.C')

    # Create plot of analytical solution
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_exact)
    plt.colorbar()
    plt.title('Exact solution for N=' + str(N) + '\nwith ' + str(BC_type) + ' B.C')

    # Create plot of difference between the two solutions
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, np.abs(u_exact - numerical_solution))
    plt.colorbar()

    """
    plt.plot(np.abs(u_exact - numerical_solution))"""

    plt.title('Error for N=' + str(N) + '\nwith ' + str(BC_type) + ' B.C.')

    if save:
        save_name = BC_type + "_N=" + str(N) + "_with_" + str(BC_type) + "_BC"
        plt.savefig(save_name + ".pdf")
    plt.show()


def meshplot(N_list, nCols=3, save=False):
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


    fig, axs = plt.subplots(nRows, nCols, figsize=(21, 7 * nRows))
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
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                # give the plot a title
                ax.set_title('Mesh for N= ' + str(N))
    # adjust
    plt.subplots_adjust(hspace=0.3, wspace=0.45)

    if save:
        plt.savefig("Mesh_for_N.pdf")
    plt.show()


if __name__ == "__main__":
    # save the plot as pdf?
    save = False
    # list of N to plot for
    N_list = [500, 1000, 2000]
    # make a meshplot
    meshplot(N_list, save=save)

