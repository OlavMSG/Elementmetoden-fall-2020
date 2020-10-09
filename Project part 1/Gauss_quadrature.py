# -*- coding: utf-8 -*-
"""
Created on 29.09.2020

@author: Olav Milian
"""
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import dblquad


def quadrature1D(a, b, Nq, g, line_int=False):
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
    # check if we have a line integral
    if line_int:
        try:
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
            I = abs_r_t * np.sum(rho_q * g2(z_q))
        except TypeError:
            raise TypeError("a and b must be list or tuple for a line integral")
    else:
        I = (b - a) / 2 * np.sum(rho_q * g(0.5 * (b - a) * z_q + 0.5 * (b + a)))
    return I

def lineintegral(a, b, Nq, g):
    # function to handle the lineintegral with local basis functions implemented
    # Weights and gaussian quadrature points
    z_q, rho_q = roots_legendre(Nq)
    a = np.asarray(a)
    b = np.asarray(b)
    # parameterization of the line C between a and b,
    # r(t) = (1-t)a/2 + (1+t)b/2,
    x = lambda t: ((1 - t) * a[0] + (1 + t) * b[0]) / 2
    y = lambda t: ((1 - t) * a[1] + (1 + t) * b[1]) / 2
    # r'(t) = -a/2+ b/2 = (b-a)/2, |r'(t)| = norm(b-a)
    abs_r_t = np.linalg.norm(b - a, ord=2) / 2
    # g times local basis
    g1 = lambda t: g(x(t), y(t)) * (1 + t) / 2  # for a
    g2 = lambda t: g(x(t), y(t)) * (1 - t) / 2  # for b
    # int_C g(x, y) * phi(x, y) ds = int_{-1}^1 g(r(t)) * phi(r(t)) |r'(t)| * 2 dt
    # = norm(b-a)  int_{-1}^1 g(r(t)) * phi(r(t)) dt
    I1 = abs_r_t * np.sum(rho_q * g1(z_q))  # load for a
    I2 = abs_r_t * np.sum(rho_q * g2(z_q))  # load for b
    return I1, I2


def quadrature2D(p1, p2, p3, Nq, g):
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
    if not line_int:
        I_exact = (np.e - 1) * np.e  # integral e^x dx x=1 to x=2
        g = lambda x: np.exp(x)
        a = 1
        b = 2
    if line_int:
        print("Testing line integral")
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


def test_quadrature2D(Itegrate_exact=False):
    # Testing:
    g = lambda x, y: np.log(x + y)

    I_exact = 1.16542
    if Itegrate_exact:
        print("Using scipy.integrate.dblquad to get I_exact")
        # y goes from 0.5*x-0.5 to x-1 as x goes from 1 to 3
        gfunc = lambda x: 0.5 * x - 0.5
        hfunc = lambda x: x - 1
        I_exact = dblquad(g, 1, 3, gfunc, hfunc)[0]

    p1, p2, p3 = [1, 0], [3, 1], [3, 2]

    Nq_list = [1, 3, 4]
    for Nq in Nq_list:
        I = quadrature2D(p1, p2, p3, Nq, g)
        print("Nq = " + "{:}".format(Nq))
        print("Using the " + "{:}".format(Nq) + "-rule I = " + "{:}".format(I))
        print("The true value of I is " + "{:}".format(I_exact))
        print("The abs. error is " + "{:}".format(np.abs(I - I_exact)))
        print("-" * 40)

if __name__ == "__main__":
    test_quadrature1D()
    test_quadrature1D(line_int=True)
    print("*"*40)
    test_quadrature2D()
    test_quadrature2D(True)
