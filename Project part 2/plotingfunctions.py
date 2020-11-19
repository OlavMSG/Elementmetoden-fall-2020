# -*- coding: utf-8 -*-
"""
Created on 16.11.2020

@author: Olav Milian
"""
import numpy as np
import matplotlib.pyplot as plt
from getplate import getPlate


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
    plt.title("$u_h(t, x,y)$ at $t=" + "{:.2f}".format(t0) + "$ \nfor $N=" +str(N) + "$")

    # Create plot of analytical solution
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h1, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + "{:.2f}".format(t1) + "$ \nfor $N=" +str(N) + "$")

    # Create plot of absolute difference between the two solutions
    plt.subplot(1, 3, 3)
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, tri, u_h2, levels=levels, extend='both')
    plt.colorbar()
    plt.title("$u_h(t, x,y)$ at $t=" + "{:.2f}".format(t2) + "$ \nfor $N=" +str(N) + "$")

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
    plt.subplots_adjust(hspace=0.3, wspace=0.4)

    # save the plot?
    if save:
        plt.savefig("plot\Mesh_for_Ns.pdf")
    plt.show()


def plotError(N_list, error_dict, time_stamps, save_name, label_txt, save=False):
    h_vec = np.array([1 / N for N in N_list])
    plt.figure("error", figsize=(14, 7))
    for key in error_dict:  # key is a int
        error = error_dict[key]
        t = np.round(time_stamps[key], 2)
        p = np.polyfit(np.log(h_vec), np.log(error), deg=1)
        print("Got convergence of order ", p[0], "for t =", t)
        label = label_txt + "$(t=" + str(t) + ")$,\nconv.order = $" +"{:.3f}".format(p[0]) +"$"
        plt.loglog(h_vec, error, 'o-', label=label, basex=2)
        # err2 = np.exp(p[0] * np.log(h_vec) + p[1])
        # plt.loglog(h_vec, err2)
    plt.xlabel("Element size, $h$")
    plt.ylabel("log " + label_txt)
    plt.gca().invert_xaxis()
    plt.grid()
    plt.title("Error Estimate using " + label_txt)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.13), ncol=3)
    if save:
        plt.savefig("plot/" + save_name +".pdf", bbox_inches='tight')
    plt.show()


def plottime(N_list, save_name, time_vec1, time_vec2=None, save=False):
    h_vec = np.array([1 / N for N in N_list])
    plt.figure("time", figsize=(12, 7))
    plt.subplot(111)
    if time_vec2 is not None:
        plt.plot(h_vec, time_vec2, 'o-', label="Time to find error estimate")
    plt.plot(h_vec, time_vec1, 'o-', label="Time to solve the problem")
    plt.xlabel("Element size, $h$")
    plt.ylabel("Time, $t$")
    plt.gca().invert_xaxis()
    plt.grid()
    plt.title("Element size v. Time")
    # adjust
    plt.subplots_adjust(hspace=0.4)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.13), ncol=3)
    if save:
        plt.savefig("plot/Time" + save_name + ".pdf", bbox_inches='tight')
    plt.show()



