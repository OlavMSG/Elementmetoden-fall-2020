# -*- coding: utf-8 -*-
"""
Created on 16.11.2020

@author: Olav Milian
"""
from getplate import getPlate
import numpy as np

def h_finder(N_list, ifprint=True):
    h_vec = []
    for N in N_list:
        p, tri, edge = getPlate(N+1)
        #   p		Nodal points, (x,y)-coordinates for point i given in row i.
        #   tri   	Elements. Index to the three corners of element i given in row i.
        #   edge  	Edge lines. Index list to the two corners of edge line i given in row i.
        h_list = []

        for nk in tri:
            # nk : node-numbers for the k'th triangle
            # the points of the triangle
            p1 = p[nk[0], :]
            p2 = p[nk[1], :]
            p3 = p[nk[2], :]

            # find distance between the nodes
            h1 = np.linalg.norm(p1 - p2)
            h2 = np.linalg.norm(p1 - p3)
            h3 = np.linalg.norm(p2 - p3)
            # append
            h_list.append(h1)
            h_list.append(h2)
            h_list.append(h3)

        # let h be the max
        h = np.max(h_list)
        h_vec.append(h)
        if ifprint:
            print("N =", N, "gives h =", h)
    if not ifprint:
        h_vec = np.asarray(h_vec)
        return h_vec


if __name__ == "__main__":

    h_wanted_list = 1 / 2**np.array([0, 1, 2, 3, 5])
    print("h =", h_wanted_list)
    # choose largest N that gives h.
    print("h =", h_wanted_list[0])
    N_list = np.arange(1, 100) # choose 12
    h_finder(N_list)

    # choose largest N that gives h.
    print("h =", h_wanted_list[1])
    N_list = [22, 23, 24, 25]  # choose 24
    h_finder(N_list)

    # choose largest N that gives h.
    print("h =", h_wanted_list[2])
    N_list = [35, 36, 37]  # choose 36
    h_finder(N_list)

    # choose largest N that gives h.
    print("h =", h_wanted_list[3])
    N_list = [64, 65, 66, 67]  # choose 66
    h_finder(N_list)

    # choose largest N that gives h.
    print("h =", h_wanted_list[4])
    N_list = [128, 129, 130]  # choose 129
    h_finder(N_list)
    print("-"*40)

