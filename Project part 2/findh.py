# -*- coding: utf-8 -*-
"""
Created on 16.11.2020

@author: Olav Milian
"""
from getdisc import GetDisc
import numpy as np



N = 7
def h_finder(N_list):
    for N in N_list:
        p, tri, edge = GetDisc(N)
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
            h2 = np.linalg.norm(p1 - p2)
            h3 = np.linalg.norm(p1 - p2)
            # append
            h_list.append(h1)
            h_list.append(h2)
            h_list.append(h3)

        # let h be the mean
        h = np.mean(h_list)
        print("N =", N, "gives h =", h)


h_wanted_list = 1 / 2**np.array([0, 1, 2, 3, 5])
print("h =", h_wanted_list)
# choose largest N that gives h.
print("h =", h_wanted_list[0])  # choose 7
N_list = [6, 7, 8]
h_finder(N_list)

# choose largest N that gives h.
print("h =", h_wanted_list[1])
N_list = [22, 23, 24, 25, 26, 27]  # choose 23
h_finder(N_list)

# choose largest N that gives h.
print("h =", h_wanted_list[2])
N_list = [77, 78, 79, 80]  # choose 78
h_finder(N_list)

# choose largest N that gives h.
print("h =", h_wanted_list[3])
N_list = [273, 274, 275]  # choose 274
h_finder(N_list)

# choose largest N that gives h.
print("h =", h_wanted_list[4])
N_list = [4070, 4071, 4072, 4073]  # choose 4071 or 4072
h_finder(N_list)