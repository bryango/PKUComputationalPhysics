#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
from toolkit.generic import print_matrix, matrix_dot
from toolkit.qrdecomp import qr_householder, qr_givens
import random
import timeit


def random_matrices(dim=6, min_max=(-1, 1), sample_size=20, seed=None):
    """ Create random matrices for testing;
        :param min_max: range of matrix elements;
        :return: list of random matrices, length = sample_size.
    """
    random.seed(seed)
    error_reporter = "Random matrix generator: "

    if len(min_max) != 2 or min_max[0] > min_max[1]:
        raise ValueError(error_reporter + "invalid random range. ")
    elif type(dim) != int or dim < 0:
        raise ValueError(error_reporter + "invalid dimension.")
    elif type(dim) != int or sample_size < 0:
        raise ValueError(error_reporter + "invalid sample size.")
    return [ [ [ random.uniform(min_max[0], min_max[1])
                 for _ in range(dim) ]
               for _ in range(dim) ]
             for _ in range(sample_size) ]


def qr_test():
    test_mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print('Test matrix A:')
    print_matrix(test_mat)
    print('')

    def method_test(qr_method, display_name):
        print(f'{display_name}:')
        q, r = qr_method(test_mat, matrix_check='notChecked')
        print('Q:')
        print_matrix(q)
        print('R:')
        print_matrix(r)
        print('QR - A:')
        qr = matrix_dot(q, r)
        print_matrix([ [ qr[i][j] - test_mat[i][j]
                         for j in range(3) ]
                       for i in range(3) ])
        print('')

    method_test(qr_householder, 'Householder')
    method_test(qr_givens, 'Givens')
    print('End of test. \n')


def qr_compare(dim, sample_size=5, runs=3, re_runs=2):
    lst = random_matrices(dim, sample_size=sample_size)
    total_tests = runs * re_runs * sample_size
    time_givens = sum(timeit.repeat(
        lambda: [qr_givens(mat, matrix_check='notChecked')
                 for mat in lst],
        'from __main__ import qr_givens',
        number=runs, repeat=re_runs)) / total_tests
    time_householder = sum(timeit.repeat(
        lambda: [qr_householder(mat, matrix_check='notChecked')
                 for mat in lst],
        'from __main__ import qr_householder',
        number=runs, repeat=re_runs)) / total_tests
    return time_householder, time_givens


if __name__ == '__main__':

    print('--------------------------------')
    if False:
        print('Debugging...')
        qr_test()

    if True:
        def qr_compare_print(dim, sample_size=5, runs=3, re_runs=2):
            time_householder, time_givens \
                = qr_compare(dim, sample_size, runs, re_runs)
            print(f'{dim}'.ljust(4), '%-12.6g    %-12.6g    %-12.6g' % (
                time_householder, time_givens, time_givens / time_householder
            ))

        print('Benchmarking...')
        print('Average time:\n')
        print('n    Householder(s)  Givens(s)       ratio')
        qr_compare_print(6, sample_size=20, runs=20)
        qr_compare_print(12, sample_size=10, runs=20)
        qr_compare_print(24, sample_size=10, runs=10)
        qr_compare_print(48, sample_size=5, runs=3)
        qr_compare_print(96, sample_size=3, runs=2)
        print('')

    print('DONE!')
    print('--------------------------------')
