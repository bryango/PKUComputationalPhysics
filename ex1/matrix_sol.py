#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Generic matrix operations: useful functions and solutions
"""

from math import *


def mrange(a, b):
    """ Sensible range """
    return range(a, b + 1)


def relativeError(aPlus: float, a: float) -> float:
    return (aPlus - a) / a


def transpose(l: list) -> list:
    return [[l[j][i] for j in range(len(l))] for i in range(len(l[0]))]


def gem(m: list, b: list) -> list:
    """ Gaussian Elimination: m x = b;
        :param m: real matrix as (column_dim) * (row_dim) list;
        :param b: row_dim-vector as list;
        :return:  solution as list; return blank list if fails.
    """

    # Define dimension:
    column_dim = len(b)  # Must have: len(m) == len(b).
    row_dim = len(m[0])  # m: NOT necessarily a square.
    try:
        augmented_m = [ m[i][:] + [b[i]] for i in range(column_dim) ]
    except IndexError:
        print('GEM Error：dimension inconsistent!')  # When len(m) != len(b).

    # Gaussian elimination -> upper triangular matrix
    # Note: row has row_dim, but its element is labeled by column_index.
    #       column has column_dim, but its element is labeled by row_index.
    for column_index in range(row_dim - 1):      # -1: no need for last column.
        pivot_row = column_index                 # Start from diagonal element
        pivot_value = augmented_m[pivot_row][column_index]
        for row in range(column_index + 1, column_dim):  # Partial pivoting
            if abs(augmented_m[row][column_index]) > abs(pivot_value):
                pivot_row = row
                pivot_value = augmented_m[row][column_index]
        if pivot_row != column_index:                    # Row switch
            augmented_m[pivot_row], augmented_m[column_index] \
                = augmented_m[column_index], augmented_m[pivot_row]
        for row in range(column_index + 1, column_dim):
            try:                                         # Gaussian Elimination
                resize_factor = augmented_m[row][column_index] / pivot_value
            except ZeroDivisionError:
                print("GEM Error：matrix is singular!")
                return []
            for column in range(column_index, row_dim + 1):
                augmented_m[row][column] \
                    -= augmented_m[column_index][column] * resize_factor

    # Substitution:
    x = []
    for index in range(column_dim - 1, -1, -1):
        partial_mx = sum(
            [ x[j] * augmented_m[index][row_dim - j - 1]
              for j in range(len(x)) ]   # Partial sum of m * solved_xComponent
        )                                # x[] is reversed for convenience
        new_xComponent = (augmented_m[index][row_dim] - partial_mx) \
            / augmented_m[index][index]
        x.append(new_xComponent)
    x.reverse()                          # Reverse back.
    return x


def cholesky(m: list, b: list) -> list:
    """ Cholesky Decomposition: m x = b;
        :param m: real *positive-definite* matrix, as dim * dim (square) list;
        :param b: dim-vector as list;
        :return:  solution as list; return blank list if fails.
    """

    # Note: m must be a square matrix!
    dim = len(m)
    if dim != len(b):
        print('Cholesky Error：dimension inconsistent!')
        return []

    # Cholesky decomposition
    # Note: Don't use [[f] * n]*n, it's a trap!
    # See: https://stackoverflow.com/a/44382900
    h = [[0. for column in range(dim)] for row in range(dim)]
    h[0][0] = sqrt(m[0][0])
    for i in range(1, dim):          # Hard to ensure positive-definiteness;
        try:                         # therefore prepared for error!
            for j in range(i):
                h[i][j] = (
                    m[i][j]
                    - sum([ h[i][k] * h[j][k] for k in range(j) ])
                ) / h[j][j]
            hii_squared = m[i][i] - sum([ h[i][k] ** 2 for k in range(i) ])
            h[i][i] = sqrt(hii_squared)
        except (ValueError, ZeroDivisionError):
            print(f'Cholesky Error: h[{i:d}][{i:d}]^2 = {hii_squared:g} <= 0, '
                  'matrix ill-conditioned!')
            return []

    # Substitution:
    h_t = transpose(h)     # Nice to have real matrix! Adjoint == transpose
    y = []
    for index in range(dim):
        new_yComponent = (
            b[index]
            - sum([ y[j] * h[index][j] for j in range(len(y)) ])
        ) / h[index][index]
        y.append(new_yComponent)
    x = []
    for index in range(dim - 1, -1, -1):
        new_xComponent = (
            y[index]
            - sum([ x[j] * h_t[index][dim - j - 1] for j in range(len(x)) ])
        ) / h_t[index][index]
        x.append(new_xComponent)
    x.reverse()
    return x
