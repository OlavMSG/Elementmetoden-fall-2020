# -*- coding: utf-8 -*-
"""
Created on 29.09.2020

@author: Olav Gran
based in old code by Ruben Mustad
"""

import numpy as np
from scipy.special import roots_legendre


def quadrature1D(a, b, Nq, g, line_int=False):
    """
    Function to do a nummerical 1D integral: line_int = False
    or a lineintegral: line_int = True
    Both using Gaussian quadrature

    Parameters
    ----------
    a : float or list/tuple
        lower bound or startpoint of line in the integration.
    b : float or list/tuple
        upper bound or endpoint of line in the integration.
    Nq : int
        How many points to use in the nummerical integration, Nq-point rule.
    g : function pointer
        pointer to function to integrate.
    line_int : bool, optional
        Do we have a lineitegral. The default is False.

    Raises
    ------
    TypeError
        If line_int=True and a or b are not in acepted form/type, meaning a and b are not list or tuple

    Returns
    -------
    I : float
        value of the integral.

    """
    # Weights and gaussian quadrature points for refrence
    # if Nq == 1:
    #     z_q = np.zeros(1)
    #     rho_q = np.ones(1) * 2
    # if Nq == 2:
    #     c = np.sqrt(1 / 3)
    #     z_q = np.array([-c, c])
    #     rho_q = np.ones(2)
    # if Nq == 3:
    #     c = np.sqrt(3 / 5)
    #     z_q = np.array([-c, 0, c])
    #     rho_q = np.array([5, 8, 5]) / 9
    # if Nq == 4:
    #     c1 = np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)
    #     c2 = np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7)
    #     z_q = np.array([-c1, -c2, c2, c1])
    #     k1 = 18 - np.sqrt(30)
    #     k2 = 18 + np.sqrt(30)
    #     rho_q = np.array([k1, k2, k2, k1]) / 36
    
    # Weights and gaussian quadrature points, also works for Nq larger than 4
    z_q, rho_q = roots_legendre(Nq)
    # check if we have a line integral, given by user
    if line_int:
        try:
            # convert a and b to numpy arrays
            a = np.asarray(a)
            b = np.asarray(b)
            # parameterization of the line C between a and b,
            # r(t) = (1-t)a/2 + (1+t)b/2,
            x = lambda t: ((1 - t) * a[0] + (1 + t) * b[0]) / 2
            y = lambda t: ((1 - t) * a[1] + (1 + t) * b[1]) / 2
            # r'(t) = -a/2+ b/2 = (b-a)/2, |r'(t)| = norm(b-a) / 2
            abs_r_t = np.linalg.norm(b - a, ord=2) / 2
            # int_C g(x, y) ds = int_{-1}^1 g(r(t)) |r'(t)| dt = norm(b-a)/2  int_{-1}^1 g(r(t)) dt
            g2 = lambda t: g(x(t), y(t))
            # compute the integral nummerically
            I = abs_r_t * np.sum(rho_q * g2(z_q))
        except TypeError:
            # raise an error if a and b are not in acepted form/type.
            raise TypeError("a and b must be list or tuple for a line integral")
    else:
        # compute the integral nummerically
        I = (b - a) / 2 * np.sum(rho_q * g(0.5 * (b - a) * z_q + 0.5 * (b + a)))
    return I


def quadrature2D(p1, p2, p3, Nq, g):
    """
    Function to do a nummerical 2D integral using Gaussian quadrature on a triangle

    Parameters
    ----------
    p1 : list/tuple
        First vertex of the triangle.
    p2 : list/tuple
        Second vertex of the triangle.
    p3 : list/tuple
        Third vertex of the triangle.
    Nq : int
        How many points to use in the nummerical integration, Nq-point rule.
    g : function pointer
        pointer to function to integrate.

    Raises
    ------
    ValueError
        If Nq is not in {1, 3, 4}.

    Returns
    -------
    I : float
        Value of the integral.

    """
    if Nq not in (1, 3, 4):
        raise ValueError("Nq is not 1, 3 or 4. Nq =" + "{.}".format(Nq))

    # Weights and gaussian quadrature points
    if Nq == 1:
        Z = np.ones((1, 3)) / 3
        rho = np.ones(1)
    if Nq == 3:
        Z = np.array([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        rho = np.ones(1) / 3
    if Nq == 4:
        Z = np.array([[1 / 3, 1 / 3, 1 / 3], [3 / 5, 1 / 5, 1 / 5], [1 / 5, 3 / 5, 1 / 5], [1 / 5, 1 / 5, 3 / 5]])
        rho = np.array([-9 / 16, 25 / 48, 25 / 48, 25 / 48])
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


def test_quadrature1D(line_int=False):
    """
    Function to test the quadrature1D function and printing info to user.

    Parameters
    ----------
    line_int : bool, optional
        Do we have a lineitegral. The default is False.

    Returns
    -------
    None.

    """
    if not line_int:
        print("-" * 40)
        print("Testing 1D integral")
        print("-" * 40)
        I_exact = (np.e - 1) * np.e  # integral e^x dx x=1 to x=2
        g = lambda x: np.exp(x)
        a = 1
        b = 2
    if line_int:
        print("-" * 40)
        print("Testing line integral")
        print("-" * 40)
        # integral of g from (0,0) to (2,2)
        # int_C g ds = sqrt(1+1) int_{0}^2 g(t, 1*t) dt
        I_exact = (np.e ** 4 - 1) / np.sqrt(2)
        g = lambda x, y: np.exp(x + y)
        a = [0, 0]
        b = [2, 2]

    for Nq in range(1, 5):
        I = quadrature1D(a, b, Nq, g, line_int=line_int)
        print("Nq = " + "{:}".format(Nq))
        print("Using the " + "{:}".format(Nq) + "-rule I = " + "{:}".format(I))
        print("The true value of I is " + "{:}".format(I_exact))
        print("The abs. error is " + "{:}".format(np.abs(I - I_exact)))
        print("-" * 40)


def test_quadrature2D():
    """
    Function to test the quadrature2D function and printing info to user.

    Returns
    -------
    None.

    """
    # Testing:
    g = lambda x, y: np.log(x + y)

    I_exact = 1.16542

    p1, p2, p3 = [1, 0], [3, 1], [3, 2]

    Nq_list = [1, 3, 4]
    print("-" * 40)
    print("Testing 2D integral")
    print("-" * 40)
    for Nq in Nq_list:
        I = quadrature2D(p1, p2, p3, Nq, g)
        print("Nq = " + "{:}".format(Nq))
        print("Using the " + "{:}".format(Nq) + "-rule I = " + "{:}".format(I))
        print("The true value of I is " + "{:}".format(I_exact))
        print("The abs. error is " + "{:}".format(np.abs(I - I_exact)))
        print("-" * 40)

def Gauss_quadrature_tester():
    """
    Function to run all the test in this script.

    Returns
    -------
    None.

    """
    test_quadrature1D()
    test_quadrature1D(line_int=True)
    test_quadrature2D()

if __name__ == "__main__":
    Gauss_quadrature_tester()
