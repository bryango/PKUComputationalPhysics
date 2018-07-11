#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Determinant of Hilbert matrix
"""

from math import *
import sympy as sym
import pandas as pd
from matrix_sol import mrange, relativeError, transpose
sym.init_printing(wrap_line=False, pretty_print=False)


def hilbertMatrix(n: int):
    """ Construct Hilbert matrix, using exact number """
    return sym.Matrix(
        [[sym.Integer(1) / sym.Integer(i + j - 1)
          for i in mrange(1, n)]
         for j in mrange(1, n)])


hilbertDetExact = [hilbertMatrix(n).det() for n in mrange(1, 10)]
hilbertDetLogRef = [log(item) for item in hilbertDetExact]


def logC(n: int) -> float:
    """ Calculate log c_n; log = ln, i.e. base e """
    return sum([log(factorial(m)) for m in mrange(1, n - 1)])


hilbertDetLogTheoretical = [4 * logC(n) - logC(2 * n) for n in mrange(1, 10)]
hilbertDetLogAsymptotic = [log(4**(- n**2)) for n in mrange(1, 10)]


def integralApprox(k: int) -> float:
    """ Estimate the sum of m*log(m), by integration """
    return 1 / 8 * (4 - 9 * log(3) - 2 * k * (1 + k) * (1 + log(4))
                    + log(256) + (1 + 2 * k)**2 * log(1 + 2 * k))


def hilbertDetLogApproxFunction(n: int) -> float:
    """ Full form: Stirling approximation of det(H_n) """
    return 4 * integralApprox(n - 1) - integralApprox(2 * n - 1) \
        + (2 * n - 1) * log(n - 1) - (n - 1 / 4) * log(2 * n - 1) \
        + (n - 9 / 4) * log(2 * pi) + 3 / 2


hilbertDetLogApprox = ['N/A'] \
    + [hilbertDetLogApproxFunction(n) for n in mrange(2, 10)]

hilbertDetLogData = pd.DataFrame(
    transpose([hilbertDetLogRef, hilbertDetLogTheoretical,
               hilbertDetLogApprox, hilbertDetLogAsymptotic]),
    columns=['Exact', 'Theoretical', 'Nice asymptotic', 'Rough asymptotic'])
hilbertDetLogData.index = list(mrange(1, 10))

hilbertDetLogError = pd.DataFrame(
    [[relativeError(hilbertDetLogTheoretical[n], hilbertDetLogRef[n]),
      relativeError(hilbertDetLogApprox[n], hilbertDetLogRef[n]),
     relativeError(hilbertDetLogAsymptotic[n], hilbertDetLogRef[n])]
     for n in range(0 + 1, 10)],
    columns=['Theoretical', 'Nice asymptotic', 'Rough asymptotic'])
hilbertDetLogError.index = list(mrange(2, 10))


def hilbertDetLogFunction(n: int) -> float:
    return 4 * logC(n) - logC(2 * n)


largeNs = [100, 1000, 10000]
# WARNING: This will take a loooong time! Don't run unless prepared!
# hilbertDetLogTheoretical_largeNs = {
#     f'{n}': 4 * logC(n) - logC(2 * n) for n in largeNs
# }
hilbertDetLogTheoretical_largeNs = {
    '100': -13680.745699832958,
    '1000': -1384458.6494934857,
    '10000': -138611060.08241153
}

hilbertDetLogError_largeN = pd.DataFrame(
    [[relativeError(hilbertDetLogApproxFunction(n),
                    hilbertDetLogTheoretical_largeNs[f'{n}']),
      relativeError(- n**2 * log(4),
                    hilbertDetLogTheoretical_largeNs[f'{n}'])]
     for n in largeNs],
    columns=['Nice asymptotic', 'Rough asymptotic'])
hilbertDetLogError_largeN.index = largeNs
