#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from toolkit.generic import mrange
from toolkit.integral import integrate_romberg
from math import exp, pi, factorial, sqrt, floor
ee = exp(1)


def n_square_generator(n_max, opt=None):
    """ Generate n_x^2 + n_y^2 + n_z^2,  up to n_max;
        :param opt: 'nozero' will exclude origin.
    """
    n_max = floor(n_max)

    def cond(x, y, z):
        return (x != 0 or y != 0 or z != 0) if opt == 'nozero' else True

    return (x ** 2 + y ** 2 + z ** 2
            for x in mrange(-n_max, n_max)
            for y in mrange(-n_max, n_max)
            for z in mrange(-n_max, n_max) if cond(x, y, z) )


# Ref params ~ n_max_alpha=5, n_max_beta=2, k_max=15, int_order=7
# Many thanks for WZY!
def zeta(q2: float, n_max_alpha=7, n_max_beta=2, k_max=16, int_order=7):
    """ Calculate zeta_{00} (s = 1, q^2 = q2);
        :param n_max_alpha: cut-off index of the first sum_n;
        :param n_max_beta: cut-off index of the second sum_n (nozero);
        :param k_max: cut-off index of sum_k;
        :param int_order: Romberg integral order.
    """

    try:
        sum_alpha = sum(
            ( exp(q2 - n2) / (n2 - q2)
              for n2 in sorted(n_square_generator(n_max_alpha), reverse=True) )
        )                          # Sorted summation for better accuray
    except ZeroDivisionError:
        raise ValueError(f"Zeta function singular at q2 = {q2}.")
    except ValueError:
        raise ValueError(f"Zeta complains: n_max_alpha not positive integer.")
    try:
        sum_k = sum(q2**n / (n - 1 / 2) / factorial(n)
                    for n in mrange(1, k_max))
        integral = integrate_romberg(
            lambda x: (
                sum( exp(-n2 * pi**2 / x)
                     for n2 in sorted(
                    n_square_generator(n_max_beta, 'nozero'), reverse=True)
                ) * exp(x * q2) * x**(-3 / 2)
            ), 0, 1, interpol_order=int_order)
    except ValueError:
        raise ValueError(f"Integration failed for q2 = {q2}")
    return pi * integral / sqrt(4 * pi) \
        + pi / 2 * sum_k \
        + sum_alpha / sqrt(4 * pi) - pi


def eps(q2: float, n_max_alpha=7, n_max_beta=2, k_max=16, int_order=7,
        opt=None):
    """ Estimate zeta error `epsilon`;
        Parameters follow those of `zeta`;
        :return: [ overall `eps_sum`, dict_of_each_contribution ]
            if opt='detailed'.
    """

    eps_k = 1 / ((2 * k_max + 1) * sqrt(2 * (k_max + 1) / pi)) \
        * (ee * q2 / (k_max + 1))**k_max
    eps_alpha_scaled = sqrt(pi / 4) * exp(1. - n_max_alpha**2 / q2)
    try:
        integral = integrate_romberg(
            lambda t: exp(t * q2 - pi**2 * n_max_beta**2 / t) * t**(-3 / 2),
            0, 1, interpol_order=int_order)
    except ValueError:
        raise ValueError(f"Integration failed for q2 = {q2}")
    eps_beta_scaled = 2 * (pi * n_max_beta)**2 \
        * (n_max_beta / pi**2)**(2 / 3) * integral
    eps_sum = eps_k + eps_alpha_scaled + eps_beta_scaled
    return [ eps_sum, {
        '$k$': eps_k,
        r'$\alpha$': eps_alpha_scaled,
        r'$\beta$': eps_beta_scaled
    } ] if opt == 'detailed' else eps_sum


def phase_shift(q2):
    return (.25 * q2 + 1.) * (pi**(3 / 2))
