from math import log2
import numpy as np
import cupy
from chainer import cuda


def ispow2(x):
    return log2(x).is_integer()


# def histogram__(x, bins):
#     assert(x.ndim == bins.ndim == 1)
#     assert(ispow2(bins.size-1))
#     return histogram_(x, cupy.zeros(bins.size-1), bins, 0, bins.size-1)


def histogram__(x, result, bins, low, high):
    if x.size == 0:
        return result

    if high - low == 1:
        result[low] = x.shape[0]
        return result
    mid = low + (high - low) // 2

    result = histogram_(x[x < bins[mid]], result, bins, low, mid)
    result = histogram_(x[x >= bins[mid]], result, bins, mid, high)
    return result


def histogram(x, bins):
    y = cupy.zeros(bins.size-1, dtype=cupy.int32)
    cupy.ElementwiseKernel(
        'S x, raw T bins, int32 n_bins',
        'raw int32 y',
        """
        int high = n_bins-1;
        int low = 0;

        while(high-low > 1) {
            int mid = (int)(low + (high-low) / 2);
            if(bins[mid] <= x) {
                low = mid;
            } else {
                high = mid;
            }
        }
        atomicAdd(&y[low], 1);
        """
    )(x, bins, bins.size, y)
    return y
