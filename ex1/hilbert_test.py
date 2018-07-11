#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sympy as sym
import pandas as pd
from matrix_sol import *
from hilbert_det import hilbertMatrix

## Try to write a `class`, turns out to be a lost cause...
# class hilbert_mat:
#     """ Hilbert matrix: construction & properties """
#
#     def ln_cn(n: int) -> float:
#         """ Calculate ln c_n, to be used in hilbert_mat.det """
#         return sum([log(factorial(i)) for i in range(1, n)])
#
#     def __init__(self, dim):
#         """ Construction of Hilbert matrix with dimension dim """
#         self.dim = dim
#         self.element = [[1. / (i + j - 1)
#             for j in range(1, n + 1)]
#                 for i in range(1, n + 1)]


def hilbertSol(n: int):
    """ Calculate H.inverse dot b, b: constant vector of 1 """
    return hilbertMatrix(n).inv().dot(
        sym.Matrix(eval('[1] * n')))


# Generate standard solutions
hilbertSolList = [hilbertSol(n) for n in mrange(2, 10)]


def hilbertMatrixNumerical(n):
    return [ [ 1. / (i + j - 1)
               for i in mrange(1, n)]
             for j in mrange(1, n)]


# Create error list: index 0 for GEM, 1 for Cholesky.
errorList = [[0. for n in mrange(2, 10)] for method in range(2)]
for n in mrange(2, 10):
    m = hilbertMatrixNumerical(n)
    b = [1.] * n
    x_gem = gem(m, b)
    x_cholesky = cholesky(m, b)
    # Index handling is terrifying! mrange(2, 10) -> 0, 1, ... , 8
    errorList[0][n - 2] \
        = max([ relativeError(x_gem[i], hilbertSolList[n - 2][i])
                for i in range(n)])
    errorList[1][n - 2] \
        = max([ relativeError(x_cholesky[i], hilbertSolList[n - 2][i])
                for i in range(n)])
errorTable = pd.DataFrame(transpose(errorList), columns=['GEM', 'Cholesky'])
errorTable.index = mrange(2, 10)
errorTable
