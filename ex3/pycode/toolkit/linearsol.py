#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from toolkit.generic import mrange, transpose, matrix_check_dim
from math import sqrt
# from copy import deepcopy


class AugmentedMatrix(object):
    """ Construct the augmented matrix:
        :param m: SQUARE matrix as (dim * dim) list;
        :param b: (dim)-vector as list.
    """
    def __init__(self, m: list, b: list, history='tracked') -> list:
        """ Generated rows won't be saved if history = 'notTracked'. """

        # Get dimensions; reject bad matrix m:
        badMatrixWarning = ('Error constructing augmented matrix: '
                            'Not a matrix! Try again.')
        try:
            if len(b) != len(m):
                print('Error constructing augmented matrix: '
                      'Dimensions inconsistent!')
            else:
                self.dimension = matrix_check_dim(
                    m, spec='square', warning=badMatrixWarning
                )[0]
        except TypeError:
            raise ValueError(badMatrixWarning)

        # index = 0 before initialization
        self.index = 0
        # Set up an i-indexed generator
        # THIS IS a DEEPCOPY! No worries.
        self.data = enumerate(
            ( m[i][:] + [b[i]] for i in range(self.dimension) ), 0
        )
        # Tracked history
        self.history = None if history == 'notTracked' else []

    def iterator(self, info='data'):
        """ Returns matrix as iterator;
            unless info='index', then returns current row index. """
        return self.index if info == 'index' else self.data

    def next(self):
        self.index, self.row = next(self.data)
        if self.history is not None:
            self.history.append((self.index, self.row))
        return self.row

    def list(self, info='matrix'):
        if self.history is not None:
            self.history.extend(list(self.data))
            return self.history if info == 'indexed' else transpose(
                self.history)[1]
        else:
            print('Matrix generator history not tracked; '
                  'impossible to list matrix elements. ')

    def dim(self):
        return self.dimension


def triangular_solve(type: str, m: list, b=[]) -> list:
    """ Triangular Substitution: m x = b;
        :param m: TRIANGULAR SQUARE matrix as (dim * dim) list;
            OR, if b = [], take m as an AUGMENTED MATRIX;
        :param b: (dim)-vector as list;
        :param type: `upper` or `lower`;
        :return: solution as list; return blank list if fails.
    """

    while type not in ['lower', 'upper']:
        print('`upper` / `lower` triangular matrix? Not specified!')
        return []

    if b == []:   # We might have a augmented m; now reject bad matrix:
        badMatrixWarning = ('Failed solving triangular system: '
                            'Not an augmented matrix! Try again.')
        try:
            if len(m) != len(m[0]) - 1:
                print('Failed solving triangular system: '
                      'Dimensions inconsistent!')
            else:
                m_dim = matrix_check_dim(
                    m, warning=badMatrixWarning
                )[0]
        except TypeError:
            raise ValueError(badMatrixWarning)
    else:
        augmentedSystem = AugmentedMatrix(m, b)

    augmented_m = m if b == [] else augmentedSystem.list()
    j_dim = i_dim = m_dim if b == [] else augmentedSystem.dim()
    # Note: j_dim is the dimension of m, NOT augmented_m

    i_range = range(i_dim) if type == 'lower' else mrange(i_dim - 1, 0, -1)
    jndex = (lambda j: j) if type == 'lower' else (lambda j: j_dim - j - 1) \
                                                                # noqa: E731
    x = []
    try:
        for i in i_range:
            x.append(
                ( augmented_m[i][j_dim] - sum([
                    augmented_m[i][jndex(j)] * x[j]
                        for j in range(len(x))                  # noqa: E131
                ]) ) / augmented_m[i][i]
            )
        if type == 'upper':  # For upper type, x[] was reversed for convenience
            x.reverse()      # Now reverse it back
        return x
    except ZeroDivisionError:
        raise ValueError("Triangular substitution failed：matrix is singular!")
        return []


def gem(m: list, b: list) -> list:
    """ Gaussian Elimination: m x = b;
        :param m: REAL, SQUARE matrix as (dim) * (dim) list;
        :param b: (dim)-vector as list;
        :return:  solution as list; return blank list if fails.
    """

    # Dimensions & basics:
    augmentedInput = AugmentedMatrix(m, b)
    augmented_m = augmentedInput.list()
    i_dim = j_dim = augmentedInput.dim()

    # Gaussian elimination -> upper triangular matrix
    # Note: row has j_dim, while column has i_dim.
    for j in range(j_dim - 1):             # -1: no need for last column.
        pivot_i = j                        # Start from diagonal element
        pivot_value = augmented_m[pivot_i][j]
        for i in range(j + 1, i_dim):      # Partial pivoting
            if abs(augmented_m[i][j]) > abs(pivot_value):
                pivot_i = i
                pivot_value = augmented_m[i][j]
        if pivot_i != j:                    # Row switch
            augmented_m[pivot_i], augmented_m[j] \
                = augmented_m[j], augmented_m[pivot_i]
        for i in range(j + 1, i_dim):
            try:                           # Gaussian Elimination
                resize_factor = augmented_m[i][j] / pivot_value
            except ZeroDivisionError:
                raise ValueError("GEM Error：matrix is singular!")
                return []
            for column in range(j, j_dim + 1):
                augmented_m[i][column] \
                    -= augmented_m[j][column] * resize_factor

    # Substitution:
    return triangular_solve('upper', augmented_m)


def cholesky(m: list, b: list) -> list:
    """ Cholesky Decomposition: m x = b;
        :param m: REAL *positive-definite* matrix, as dim * dim (square) list;
        :param b: dim-vector as list;
        :return:  solution as list; return blank list if fails.
    """

    # Turn matrix m to an iterator
    augmented_m = AugmentedMatrix(m, b, 'notTracked')
    dim = augmented_m.dim()

    # Cholesky decomposition
    # Note: Don't use [[f] * n]*n, it's a trap!
    # See: https://stackoverflow.com/a/44382900
    h = [[0. for column in range(dim)] for row in range(dim)]
    # First element at top left
    h[0][0] = sqrt(augmented_m.next()[0])
    for (i, m_i) in augmented_m.iterator():
        try:                         # Hard to ensure positive-definiteness;
            for j in range(i):       # therefore be prepared for error!
                h[i][j] = (
                    m_i[j]
                    - sum([ h[i][k] * h[j][k] for k in range(j) ])
                ) / h[j][j]
            hii_squared = m_i[i] - sum([ h[i][k] ** 2 for k in range(i) ])
            h[i][i] = sqrt(hii_squared)
        except (ValueError, ZeroDivisionError):
            raise ValueError('Cholesky Error: '
                             f'h[{i:d}][{i:d}]^2 = {hii_squared:g} <= 0, '
                             'matrix ill-conditioned!')
            return []

    # Substitution:
    h_t = transpose(h)     # Nice to have real matrix! Adjoint == transpose
    y = triangular_solve('lower', h, b)
    return triangular_solve('upper', h_t, y)


def thomas_tridiagonal_solve(diag: list, upper: list, lower: list, rhs: list):
    """ Thomas algorithm for solving tridiagonal matrix;
        :param diag: diagonal elements as n-list;
        :param upper: upper diagonal elements as (n-1)-list, row-indexed
        :param lower: lower diagonal elements as n-list, row-indexed;
            Note: first entry should be zero for `lower`.
    """

    n = len(diag)
    if (len(upper) > n or len(lower) > n):
        raise ValueError('Tri-diagonal construction failed: '
                         'dimensions inconsistent!')
    if float(lower[0]) != 0.:
        print('First entry of lower diagonal elements nonzero!'
              'Information lost. ')

    # LU decomposition
    # Convention of notation: liuchuan's Numerical.pdf, ver 0.98
    alpha = [diag[0]]
    # Lower diagonal elements for L
    beta = [0.]
    for i in range(1, n):
        beta.append(lower[i] / alpha[i - 1])
        alpha.append(diag[i] - beta[i] * upper[i - 1])

    # L-triangular solve:
    # Note: Did not invoke triangular_solve(), for this is sooo sparse
    x_temp = [rhs[0]]
    for i in range(1, n):
        x_temp.append(rhs[i] - beta[i] * x_temp[i - 1])
    # U-triangular solve:
    try:
        x = [x_temp[n - 1] / alpha[n - 1]]
        for i in range(1, n):
            x.append(
                (x_temp[n - i - 1] - upper[n - i - 1] * x[i - 1])
                / alpha[n - i - 1]
            )
    except ZeroDivisionError:
        raise ValueError('Encounter singularity '
                         'when solving tridiagonal system.')
    x.reverse()
    return x


# # <codecell>
# # Hilbert matrix for debugging
# def hilbertMatrixNumerical(n):
#     return [ [ 1. / (i + j - 1)
#                for i in mrange(1, n)]
#              for j in mrange(1, n)]
#
#
# n = 7
# m = hilbertMatrixNumerical(n)
# b = [1.] * n
# hilbertSolList = [7, -336, 3780, -16800, 34650, -33264, 12012]
# x_gem = gem(m, b)
# x_cholesky = cholesky(m, b)
# print(x_gem[:3])
# print(x_cholesky[:3])
# print(m == hilbertMatrixNumerical(n))
