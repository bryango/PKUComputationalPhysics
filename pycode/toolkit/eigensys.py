#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
from math import log10
from copy import deepcopy
from toolkit.generic import (
    matrix_check_dim, relative_error,
    xtranspose, vec_square, normalize_max
)


def max_eigensystem(coefficient_matrix, accuarcy=10**-16, max_runs=10,
                    notify=True):
    """ Find maximum eigenvalue & corresponding eigenvector,
        using power iteration.
        :param max_runs: max attempts,
            for random generations of initial eigenvector
        :return: (eigenvalue, eigenvector)
    """
    a_mat = deepcopy(coefficient_matrix)
    n = matrix_check_dim(a_mat, spec='square')[0]

    # Initial eigenvalue
    v = 0.
    attempts = 0
    stuck = False
    ever_stuck = False

    # Iteration starts from a random real vector
    # Regenerate! If eigenvalue v falls back to 0,
    # ... or stuck in loop
    while attempts < max_runs and any([
        round(v, int(-log10(accuarcy))) == 0,
        stuck is True
    ]):
        attempts += 1

        # Notation: liuchuan's Numerical.pdf, ver 0.98
        z_vec = [ random.uniform(-1, 1) for _ in range(n) ]
        q_vec = normalize_max(z_vec)
        # Arbitrary `diff` to kickstart iteration
        diff = 10 * accuarcy

        # timeout: give up when iteration does not converge
        timeout = time.time() + 1

        while diff > accuarcy:
            q_vec0 = q_vec
            z_vec = [ sum( a_mat[i][j] * q_vec[j]
                           for j in range(n) )
                      for i in range(n)]
            q_vec = normalize_max(z_vec)

            # Relative error,
            # ... by maximun difference in components
            diff = max(map(
                lambda x: abs(relative_error(*x)),
                xtranspose([q_vec, q_vec0])
            ))
            if time.time() > timeout:
                stuck = True
                ever_stuck = True
                if notify:
                    print('`max_eigensystem`: Iteration did not converge! '
                          f'dim = {n}, Re-trying ...')
                break
            else:
                stuck = False

        # Rayleigh quotient
        v = sum(q_vec0[i].conjugate() * z_vec[i] for i in range(n)) \
            / vec_square(q_vec0)

    if attempts == max_runs:
        print(f'Total: {attempts} in max {max_runs} attempts '
              f'for `max_eigensystem`; final diff: {diff} (relatively).')
    if ever_stuck and notify:
        print('--------------------------------')
    return v, tuple(q_vec)
