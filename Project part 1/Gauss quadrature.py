# -*- coding: utf-8 -*-
"""
Created on 29.09.2020

@author: Olav Milian
"""
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import dblquad

def quadrature1D(a, b, Nq, g):
    # Weights and gaussian quadrature points
    """
    if Nq == 1:
        z_q = np.zeros(1)
        rho_q = np.ones(1) * 2
    if Nq == 2:
        c = np.sqrt(1 / 3)
        z_q = np.array([-c, c])
        rho_q = np.ones(2)
    if Nq == 3:
        c = np.sqrt(3 / 5)
        z_q = np.array([-c, 0, c])
        rho_q = np.array([5, 8, 5]) / 9
    if Nq == 4:
        c1 = np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)
        c2 = np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7)
        z_q = np.array([-c1, -c2, c2, c1])
        k1 = 18 - np.sqrt(30)
        k2 = 18 + np.sqrt(30)
        rho_q = np.array([k1, k2, k2, k1]) / 36
    """

    z_q, rho_q = roots_legendre(Nq)

    I = (b - a) / 2 * np.sum(rho_q * g(0.5 * (b - a) * z_q + 0.5 * (b + a)))
    return I


def quadrature2d(p1, p2, p3, Nq, g):

    if Nq not in (1, 3, 4):
        raise ValueError("Nq is not 1, 3 or 4. Nq =" + "{.}".format(Nq))

    """# convert p1, p2 and p3 to numpy array if they are not
    if isinstance(p1, (list, tuple)):
        p1 = np.array(p1)
    if isinstance(p2, (list, tuple)):
        p2 = np.array(p2)
    if isinstance(p3, (list, tuple)):
        p3 = np.array(p3)"""

    # Weights and gaussian quadrature points
    if Nq == 1:
        Z = np.ones((1, 3)) / 3
        rho = np.ones(1)
    if Nq == 3:
        Z = np.array([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        rho = np.ones(1) / 3
    if Nq == 4:
        Z = np.array([[1/3, 1/3, 1/3], [3/5, 1/5, 1/5], [1/5, 3/5, 1/5], [1/5, 1/5, 3/5]])
        rho = np.array([-9/16, 25/48, 25/48, 25/48])
    # determinant of the Jacobi matrix
    det_Jac = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    # area of the triangle
    Area = abs(det_Jac) / 2

    # Calculating the phyiscal points
    # mapping (xhi1, xhi2, xhi3) to (x, y) given matrix of xhi-s Z
    x = p1[0] * Z[:, 0] + p2[0] * Z[:, 1] + p3[0] * Z[:, 2]
    y = p1[1] * Z[:, 0] + p2[1] * Z[:, 1] + p3[1] * Z[:, 2]

    # Calculating the Gaussian qudrature summation formula
    I = Area * np.sum(rho * g(x, y))
    return I

def test_quadrature1D():

    g = lambda x: np.exp(x)
    I_exact = (np.e - 1) * np.e  # integral e^x dx x=1 to x=2

    a = 1
    b = 2
    for Nq in range(1, 5):
        I = quadrature1D(a, b, Nq, g)
        print("Nq = "+ "{:}".format(Nq))
        print("Using the " + "{:}".format(Nq) + "-rule I = " + "{:}".format(I))
        print("The true value of I is " + "{:}".format(I_exact))
        print("The abs. error is " + "{:}".format(np.abs(I - I_exact)))
        print("-"*40)

def test_quadrature2d(Itegrate_exact=False):
    # Testing:
    g = lambda x, y: np.log(x + y)

    I_exact= 1.16542
    if Itegrate_exact:
        print("Using scipy.integrate.dblquad to get I_exact")
        # y goes from 0.5*x-0.5 to x-1 as x goes from 1 to 3
        gfunc = lambda x: 0.5 * x - 0.5
        hfunc = lambda x: x - 1
        I_exact = dblquad(g, 1, 3, gfunc, hfunc)[0]

    p1, p2, p3 = [1, 0], [3, 1], [3, 2]

    Nq_list = [1, 3, 4]
    for Nq in Nq_list:
        I = quadrature2d(p1, p2, p3, Nq, g)
        print("Nq = " + "{:}".format(Nq))
        print("Using the " + "{:}".format(Nq) + "-rule I = " + "{:}".format(I))
        print("The true value of I is " + "{:}".format(I_exact))
        print("The abs. error is " + "{:}".format(np.abs(I - I_exact)))
        print("-" * 40)




if __name__ == "__main__":
    test_quadrature1D()
    test_quadrature2d()
    test_quadrature2d(True)

