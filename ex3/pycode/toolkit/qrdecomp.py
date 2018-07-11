#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from toolkit.generic import identity_matrix, matrix_check_dim
from math import sqrt
from copy import deepcopy


def householder_vector(input_vector: list, round_accuracy=16) \
        -> (float, list):
    """ Construct aHouseholder reduction,
        which move all components of `input_vector` to its 1st entry;
        :return: Householder vector v (normal vector of reflection) & its norm,
            as in (v_square, v).
    """
    # Normalize input_vector by inf-norm to prevent overflow; DEEPCOPY
    x_max = max(map(abs, input_vector))
    v = list(map(lambda x: x / x_max, input_vector))
    # Euclidean norm
    res_square = sum(x_i**2 for x_i in v[1:])
    x_norm = sqrt(res_square + v[0]**2)

    if round(res_square, round_accuracy) != 0:
        if v[0] <= 0:
            v[0] -= x_norm  # Minus so that end result is positive
        else:
            v[0] = - res_square / (v[0] + x_norm)  # Better accuracy
        v_square = v[0]**2 + res_square
        return (v_square, v)
    else:
        return (0., [0.] * len(v))


def qr_householder(matrix_input: list,
                   round_accuracy=16,
                   matrix_check='notChecked') -> (list, list):
    """ QR decomposition by Householder;
        :return: Q, R matrices.
    """
    r_mat = deepcopy(matrix_input)
    n = matrix_check_dim(r_mat, check=matrix_check, spec='square')[0]

    v_list = []  # List of Householder vectors, as in (v_square, v)
    q_mat = identity_matrix(n)  # Initialize Q as identity matrix

    # k as column index
    for k in range(n - 1):
        # Take diagonal, and elements below
        v_square, v = householder_vector(
            [ r_i[k] for r_i in r_mat[k:] ], round_accuracy
        )
        v_list.append((v_square, v))

        vt_dot_r = [ sum(v[l - k] * r_mat[l][j] for l in range(k, n))
                     for j in range(k, n) ]
        for i in range(k, n):
            coefficient = 2 * v[i - k] / v_square
            for j in range(k, n):
                r_mat[i][j] -= coefficient * vt_dot_r[j - k]

    for v_square, v in reversed(v_list):
        k = n - len(v)
        v_dot_q = [ sum(v[l - k] * q_mat[l][j] for l in range(k, n))
                    for j in range(k, n) ]
        for i in range(k, n):
            coefficient = 2 * v[i - k] / v_square
            for j in range(k, n):
                q_mat[i][j] -= coefficient * v_dot_q[j - k]

    return q_mat, r_mat


def qr_givens(matrix_input: list,
              round_accuracy=16,
              matrix_check='notChecked') -> (list, list):
    """ QR decomposition by Givens;
        :return: Q, R matrices.
    """
    r_mat = deepcopy(matrix_input)
    n = matrix_check_dim(r_mat, check=matrix_check, spec='square')[0]

    q_mat = identity_matrix(n)
    for i in range(1, n):
        for j in range(i):
            # Eliminate (i, j), for j < i
            if round(r_mat[i][j], round_accuracy) != 0.:
                smaller, larger = sorted(
                    map(abs, (r_mat[j][j], r_mat[i][j]))
                )
                scale = larger * sqrt(1 + (smaller / larger) ** 2)

                c = r_mat[j][j] / scale  # Cosine
                s = r_mat[i][j] / scale  # Sine

                # Previous iterations have ensured that for k < j < i,
                #   r_mat[i][k] = 0, r_mat[j][k] = 0.
                # Therefore, summations need not go all the way from 1 to n.
                for k in range(j, n):
                    r_mat[j][k], r_mat[i][k] \
                        = c * r_mat[j][k] + s * r_mat[i][k], \
                        - s * r_mat[j][k] + c * r_mat[i][k]
                for k in range(i + 1):
                    q_mat[k][j], q_mat[k][i] \
                        = c * q_mat[k][j] + s * q_mat[k][i], \
                        - s * q_mat[k][j] + c * q_mat[k][i]
    return q_mat, r_mat
