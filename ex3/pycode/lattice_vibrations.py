#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
from math import cos, pi
from toolkit.generic import MatrixByRule, print_vector
from toolkit.eigensys import max_eigensystem


def stiffness_matrix_element(dim: int, i: int, j: int):
    def main_components(i: int, j: int):
        if i == j:
            return 2
        elif j in (i + 1, i - 1):
            return -1
        else:
            return 0

    if isinstance(dim, int) and dim > 1:
        if (i, j) in [(0, dim - 1), (dim - 1, 0)]:
            return main_components(i, j) - 1
        elif all( index in range(dim) for index in [i, j] ):
            return main_components(i, j)
        else:
            raise IndexError('illegal index for matrix_element')
    else:
        raise IndexError('illegal dimension for matrix_element')


def lattice_max_mode(length=10, notify=True):
    n = int(length)
    mat = MatrixByRule(lambda i, j: stiffness_matrix_element(n, i, j), n)
    return max_eigensystem(mat, notify=notify)


if __name__ == '__main__':
    print('--------------------------------')
    if True:

        length = 9
        print(f'1D lattice length: {length}')
        eigenvalue, eigenvector = lattice_max_mode(length)

        print('')
        print('Max eigenvalue and its eigenvector:')
        print(
            "%.6f," % eigenvalue,
            'ref for odd dim:',
            "%.6f" % (4 * cos(pi / 2 / length)**2) )
        print_vector(eigenvector)
        print('')

    print('DONE!')
    print('--------------------------------')
