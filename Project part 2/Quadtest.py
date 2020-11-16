# -*- coding: utf-8 -*-
"""
Created on 16.11.2020

@author: Olav Milian
"""

from Gauss_quadrature import quadrature2D
import numpy as np

f = lambda x, y: 5 + 2*x + 3 *y

goal = 38 / 3
p1, p2, p3 = [1, 0], [3, 1], [3, 2]


test = quadrature2D(p1, p2, p3, 1, f)

print(test, goal, abs(test-goal))


f = lambda x, y: 5 + 2*x*x + 3 *y

goal = 58/3
p1, p2, p3 = [1, 0], [3, 1], [3, 2]


test = quadrature2D(p1, p2, p3, 3, f)

print(test, goal, abs(test-goal))

f = lambda x, y: 5 + 2*x*x + 3 *y*y

goal = 119/6
p1, p2, p3 = [1, 0], [3, 1], [3, 2]


test = quadrature2D(p1, p2, p3, 3, f)

print(test, goal, abs(test-goal))

f = lambda x, y: 5 + 2*x*x*x + 3 *y*y

goal = 369/10
p1, p2, p3 = [1, 0], [3, 1], [3, 2]


test = quadrature2D(p1, p2, p3, 4, f)

print(test, goal, abs(test-goal))

f = lambda x, y: 5 + 2*x*x*x + 3 *y*y*y

goal = 379/10
p1, p2, p3 = [1, 0], [3, 1], [3, 2]

test = quadrature2D(p1, p2, p3, 4, f)

print(test, goal, abs(test-goal))

f = lambda x, y: 5 + 2*x*x*x*x + 3 *y*y*y*y

goal = 1262/15
p1, p2, p3 = [1, 0], [3, 1], [3, 2]

test = quadrature2D(p1, p2, p3, 4, f)

print(test, goal, abs(test-goal))

def h(N):
    return np.floor(np.sqrt(N / np.pi))

print("-"*20)
print(h(8))