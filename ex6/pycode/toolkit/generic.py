#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Generic utilities: nice shortcuts & more!
"""

import os
import sys
import inspect
import builtins
import copy
import numpy as np
# from math import *


# <codecell>
# System settings
def path_append(path=None):
    if path is not None:
        sys.path.append(path)
    return sys.path


def abs_file_path(file_path: str):
    """ Retrieve absolute file path, relative to current FILE """
    return os.path.dirname(
        os.path.abspath(
            inspect.getfile(
                inspect.currentframe()
            ))) + file_path


# <codecell>
# List operations
def max_depth(l: list):
    """ Maximum depth of a list, found recursively. """
    return isinstance(l, list) and max(map(max_depth, l)) + 1


def mrange(a, b, h=1):
    """ Sensible (math-like) range: a, a + h, ... , b.
        Defined as range(a, b +/- 1, h) with h >/< 0.
    """
    if h > 0:
        return range(a, b + 1, h)
    elif h < 0:
        return range(a, b - 1, h)
    else:
        raise TypeError('Stepsize must be non-zero integer!')


def grid_generator(start, end, diff, always_include_end=True):
    """ Generate math-like grid using `np.linspace`,
        including endpoint (default).
    """
    if (end - start) / diff <= 0:
        raise ValueError('inconsistent stepsize & endpoint'
                         'for `grid_generator`!')
    block_count = int(
        (end - start) / abs(diff)
    )
    end_sharp = start + block_count * diff
    grid_sharp = np.linspace(start, end_sharp, block_count + 1)
    if end == end_sharp or (not always_include_end):
        return grid_sharp
    else:
        return np.append(grid_sharp, end)


# <codecell>
# Error analysis
def relative_error(aPlus: float, a: float) -> float:
    """ Relative error (aPlus - a) / a """
    return (aPlus - a) / a


# <codecell>
# Matrix operations
def xtranspose(l: list) -> map:
    """ Transpose a matrix; returns a map (memory-saving). """
    try:
        if len(set(map(len, l))) != 1:
            print('xtranspose: Not a matrix! Some elements lost.')
        return map(list, zip(*l))
    except TypeError:
        print('Cannot transpose: Not a matrix!')


def transpose(l: list) -> list:
    """ Transpose a matrix; returns a list. """
    return list(xtranspose(l))


def matrix_dot(a, b):
    """ Matrix dot product """
    try:
        m, p = matrix_check_dim(a)
        p, n = matrix_check_dim(b)
        return [ [ sum(a[i][k] * b[k][j] for k in range(p))
                   for j in range(n) ]
                 for i in range(m) ]
    except (IndexError, TypeError):
        raise ValueError("Matrix dot: invalid input!")


def print_matrix(mat):
    print(np.matrix(mat))


def print_vector(vec):
    print(np.array(vec))


def identity_matrix(dim: int):
    return [ [ 1 if i == j else 0
               for i in range(dim) ]
             for j in range(dim)]


def matrix_check_dim(input_matrix: list, spec='rectangular',
                     check='Checked', warning=None):
    """ Check matrix and get its dimensions;
        print warning and return None if `input_matrix` is bad.
    """
    if isinstance(input_matrix, MatrixByRule):
        return [ input_matrix.dimension() ] * 2

    if check == 'Checked':
        if warning is None:
            badMatrixWarning = 'Bad matrix! Check input!'
        else:
            badMatrixWarning = warning

        if len(set(map(len, input_matrix))) != 1:
            print(badMatrixWarning)
            return None
        if spec == 'square':
            if len(input_matrix) != len(input_matrix[0]):
                print('Not a SQUARE matrix (as required)!')
                return None

    return (len(input_matrix), len(input_matrix[0]))


class ListByRule(object):
    """ Construct list according to a certain rule (function);
        :param func: input function, with argument i as list index;
        :param length: length of the list.
        SLICING IS NOT SUPPORTED!
    """
    def __init__(self, func, dim: int):
        self.dim = dim
        self.func = func

    def dimension(self):
        return self.dim

    def print(self):
        return print_matrix(self.list())

    def rule(self, i):
        i_reduced = i % self.dim
        if i_reduced in range(self.dim):
            return self.func(i_reduced)
        else:
            raise IndexError('illegal index for ListByRule')

    def __getitem__(self, key):
        return self.rule(key)

    def list(self):
        return [ self.rule(i) for i in range(self.dim) ]


class MatrixByRule(ListByRule):
    """ Construct SQUARE matrix, according to a certain rule (function);
        :param func: input function, with argument (i, j);
        :param dim: dimension of the matrix.
        SLICING IS NOT SUPPORTED!
        P.S. `__init__`, `dim` & `print` inherited from `ListByRule`.
    """
    def rule(self, i, j):
        entry = (i_reduced, j_reduced) = [ x % self.dim for x in [i, j] ]
        if all( item in range(self.dim) for item in entry ):
            return self.func(*entry)
        else:
            raise IndexError('illegal index for MatrixByRule')

    def __getitem__(self, key):
        return ListByRule(lambda i: self.rule(key, i), self.dim)

    def list(self):
        return [ [self.rule(i, j) for j in range(self.dim)]
                 for i in range(self.dim) ]


def len(obj):
    if isinstance(obj, (ListByRule, MatrixByRule)):
        return obj.dimension()
    else:
        return builtins.len(obj)


def deepcopy(x, **kwargs):
    if isinstance(x, (ListByRule, MatrixByRule)):
        return x
    else:
        return copy.deepcopy(x, **kwargs)


def vec_square(vec: list):
    vec_imag = list(map(lambda x: x.imag, vec))
    if vec_imag != [0] * len(vec):
        vec_real_params = list(map(lambda x: x.real, vec))
        vec_real_params.extend(vec_imag)
    else:
        vec_real_params = vec

    x_max = max(map(abs, vec_real_params))
    return x_max**2 * sum(
        map(lambda x: (x / x_max)**2, vec_real_params)
    )


def normalize_max(vec: list):
    """ Normalize a (complex) vector, by its max (real) parameter """
    vec_imag = list(map(lambda x: x.imag, vec))
    if vec_imag != [0] * len(vec):
        vec_real_params = list(map(lambda x: x.real, vec))
        vec_real_params.extend(vec_imag)
    else:
        vec_real_params = vec

    param_max = max(map(abs, vec_real_params))
    try:
        return [x / param_max for x in vec]
    except ZeroDivisionError:
        return [0. for _ in range(len(vec))]
