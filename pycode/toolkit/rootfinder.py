#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from toolkit.generic import mrange
# from math import exp, pi, factorial, sqrt


def find_root_secant(func, x1: float, x2: float, accuracy=10**(-8)):
    """ Find root using secant method;
        :param func: equation to solve, as in f(x) = 0;
        :param x1, x2: initial pts of iteration;
        :param accuracy: goal of accuracy.
    """

    x_len = x2 - x1
    try:
        f1 = func(x1)
        f2 = func(x2)
        while abs(x_len) >= accuracy:
            x_len = (x2 - x1) * f2 / (f2 - f1)
            x1, f1 = x2, f2
            x2 -= x_len
            f2 = func(x2)
    except (ArithmeticError, ValueError):
        raise ValueError("Secant method failed to find root.")
    return x2
