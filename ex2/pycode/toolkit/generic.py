#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Generic utilities: nice shortcuts & more!
"""

# from math import *
import sys


def path_append(path=None):
    if path is not None:
        sys.path.append(path)
    return sys.path


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


def relative_error(aPlus: float, a: float) -> float:
    """ Relative error (aPlus - a) / a """
    return (aPlus - a) / a


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
