from functools import lru_cache
from math import log2

from numpy.testing import assert_array_equal
import numpy as np


def ispow2(x):
    return log2(x).is_integer()


def histogram(x, bins):
    assert(np.ndim(x) == 1)
    assert(ispow2(len(bins)-1))
    return histogram_(x, np.zeros(len(bins)-1), bins, 0, len(bins)-1)


def histogram_(x, result, bins, low, high):
    if len(x) == 0:
        return result

    if high - low == 1:
        result[low] = x.shape[0]
        return result
    mid = low + (high - low) // 2

    result = histogram_(x[x < bins[mid]], result, bins, low, mid)
    result = histogram_(x[x >= bins[mid]], result, bins, mid, high)
    return result

