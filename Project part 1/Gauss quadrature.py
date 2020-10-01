# -*- coding: utf-8 -*-
"""
Created on 29.09.2020

@author: Olav Milian
"""
# -*- coding: utf-8 -*-
"""
Created on 29.09.2020

@author: Olav Milian
"""
import numpy as np
from scipy.special import roots_legendre

def quadrature1D(a, b, Nq, g):
    z_q, rho_q = roots_legendre(Nq)

    I = (b - a) / 2 * np.sum(rho_q * g(0.5 * (b - a) * z_q + 0.5 * (b + a)))
    return I


def test_quadrature1D():

    g = lambda x: np.exp(x)
    I_goal = (np.e - 1) * np.e  # integral e^x dx x=1 to x=2

    a = 1
    b = 2
    for Nq in range(1, 5):
        I = quadrature1D(a, b, Nq, g)
        print("Using the " + "{:}".format(Nq) + "-rule I = " + "{:}".format(I))
        print("The true value of I is " + "{:}".format(I_goal))
        print("The abs. difference is " + "{:}".format(np.abs(I - I_goal)))
        print("-"*40)


"""test_quadrature1D()
g = lambda x: np.exp(x)
a = np.array([1, 0])
b = np.array([2, 0])
