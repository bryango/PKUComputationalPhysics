#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from toolkit.generic import mrange
from toolkit.interpol import neville_interpolation
# from math import exp, pi, factorial, sqrt


def integrate_romberg(func, x_min: float, x_max: float,
                      interpol_order=8):
    """ Romberg's method for integration,
        :param interpol_order: Neville interpolation order.
    """

    # Get minimal stepsize
    h0 = (x_max - x_min) / 2**interpol_order
    if h0 < 0:
        raise ValueError("x_max is smaller than x_min")
    h_list = [(x_max - x_min) / 2**i for i in range(interpol_order + 1)]

    def boundary_mod(boundary_side, boundary_pt, h):
        if boundary_side == 'max':
            return boundary_pt - h
        elif boundary_side == 'min':
            return boundary_pt + h

    def singularity_detected(boundary_side, boundary_pt):
        try:
            func(boundary_pt)
        except (ArithmeticError, ValueError):
            return True
        else:
            return False

    x_max_singular, x_min_singular = map(
        lambda boundary_info: singularity_detected(*boundary_info),
        [ ['max', x_max], ['min', x_min] ]
    )

    def new_domain_list(boundary_side, singularity_detected, boundary_old):
        """ Create a series of new endpoints, as a function of h """
        if singularity_detected:
            return [ boundary_mod(boundary_side, boundary_old, h)
                     for h in h_list ]
        else:
            return [ boundary_old ] * (interpol_order + 1)

    x_max_list, x_min_list = map(
        lambda boundary_change: new_domain_list(*boundary_change),
        [ ['max', x_max_singular, x_max],
          ['min', x_min_singular, x_min] ]
    )

    # Get all values up and ready
    try:
        # First & Last entry: placeholder for endpoints
        func_value_list = ([0.]
            + [func(x_min + i * h0)
                for i in range(1, 2**interpol_order)]
            + [0.])                                       # noqa: E128
        func_x_min, func_x_max = map(
            lambda l: list(map(func, l)),
            [x_min_list, x_max_list])
    except (ArithmeticError, ValueError):
        raise ValueError("The function has one or more singularities.")

    # T(h), by naive trapezoidal rule
    integral_value_list \
        = [h_list[i] * (
            (func_x_min[i] + func_x_max[i]) / 2
            + sum(func_value_list[ 2**(interpol_order - i) * j ]
                  for j in range(1, 2**i))
        ) for i in range(interpol_order + 1)]
    interp_integral \
        = neville_interpolation(
            list(map(lambda x: x**2, h_list)), integral_value_list
        )
    return interp_integral(0)
