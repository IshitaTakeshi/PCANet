# We recommend you to see [the original paper](https://arxiv.org/abs/1404.3606)
# before reading the code below.

# TODO use sphinx
import itertools

import numpy as np
from sklearn.decomposition import PCA


def steps(image_shape, filter_shape, step_shape):
    h, w = image_shape
    fh, fw = filter_shape
    sh, sw = step_shape

    ys = range(0, h-fh+1, sh)
    xs = range(0, w-fw+1, sw)
    return ys, xs


def output_shape(ys, xs):
    return len(ys), len(xs)


class Patches(object):
    def __init__(self, image, filter_shape, step_shape):
        self.image = image
        self.filter_shape = filter_shape

        self.ys, self.xs = steps(image.shape, filter_shape, step_shape)

    @property
    def patches_with_indices(self):
        """
        Yields patches with its location indices.
        Kernel visits the image from the left top into right bottom.

        Yields
        ------
        j : int
            Index of the patch along y axis
        i : int
            Index of the patch along x axis
        patch : np.ndarray
            (j, i)th patch
        """

        # The behaviour is same as below:
        # ```
        # for j, y in enumerate(self.ys):
        #     for i, x in enumerate(self.xs):
        #         yield j, i, self.image[y:y+sh, x:x+sw]
        # ```
        # But the code above does not work when the second time calling,
        # so we create a generator object every time of function call.
        fh, fw = self.filter_shape
        it = itertools.product(enumerate(self.ys), enumerate(self.xs))
        return ((j, i, self.image[y:y+fh, x:x+fw]) for (j, y), (i, x) in it)

    @property
    def patches(self):
        """
        Return patches
        """
        return np.array([p for j, i, p in self.patches_with_indices])

    @property
    def output_shape(self):
        return output_shape(self.ys, self.xs)


def heaviside_step(X):
    X[X > 0] = 1
    X[X <= 0] = 0
    return X


def normalize(X):
    """Subtract mean so that the mean of X be a zero vector"""
    return X - X.mean(axis=0, keepdims=True)


def images_to_patches(images, filter_shape, step_shape):
    """
    Each row of X represents a flattened patch
    The number of columns is N x M where
    N is the number of images and
    M is the number of patches that can be obtained from one image.
    """
    def f(image):
        X = Patches(image, filter_shape, step_shape).patches
        X = X.reshape(X.shape[0], -1)  # reshape each patch into a vector
        return normalize(X)

    return np.vstack([f(image) for image in images])


def convolution(images, filter_, filter_shape, step_shape):
    def convolution_(patches):
        L = np.empty(patches.output_shape)
        for j, i, patch in patches.patches_with_indices:
            L[j, i] = np.dot(filter_.flatten(), patch.flatten())
        return L

    it = (Patches(image, filter_shape, step_shape) for image in images)
    return np.array([convolution_(patches) for patches in it])


def convolutions(images, filters, filter_shape, step_shape):
    # TODO use numpy.vectorize
    c = [convolution(images, f, filter_shape, step_shape) for f in filters]
    return np.array(c)


def binarize(images):
    output = np.zeros(images.shape[1:3])
    for i, image in enumerate(reversed(images)):
        output += np.power(2, i) * heaviside_step(image)
    return output


def to_tuple_if_int(value):
    """
    If int is given, duplicate it and return as a 2 element tuple.
    """
    if isinstance(value, int):
        return (value, value)
    return value


class PCANet(object):
    def __init__(self,
                 filter_shape_l1, step_shape_l1, n_l1_output,
                 filter_shape_l2, step_shape_l2, n_l2_output,
                 block_shape):
        """
        Parameters
        ----------
        filter_shape_l1: int or sequence of ints
            The shape of the kernel in the first convolution layer.
        step_shape_l1: int or sequence of ints
            The shape of kernel step in the first convolution layer.
        n_l1_output:
            L1 in the original paper. The number of outputs obtained
            from a set of input images.
        filter_shape_l2: int or sequence of ints
            The shape of the kernel in the second convolution layer.
        step_shape_l2: int or sequence of ints
            The shape of kernel step in the second convolution layer.
        n_l2_output:
            L2 in the original paper. The number of outputs obtained
            from each L1 output.
        block_shape: int or sequence of ints
            The shape of each block in the pooling layer.
        """

        self.filter_shape_l1 = to_tuple_if_int(filter_shape_l1)
        self.step_shape_l1 = to_tuple_if_int(step_shape_l1)
        self.n_l1_output = n_l1_output

        self.filter_shape_l2 = to_tuple_if_int(filter_shape_l2)
        self.step_shape_l2 = to_tuple_if_int(step_shape_l2)
        self.n_l2_output = n_l2_output

        self.block_shape = to_tuple_if_int(block_shape)
        self.n_bins = None  # TODO make n_bins specifiable

        self.pca_l1 = PCA(n_l1_output)
        self.pca_l2 = PCA(n_l2_output)

    def convolution_l1(self, images):
        return convolutions(images, self.pca_l1.components_,
                            self.filter_shape_l1,
                            self.step_shape_l1)

    def convolution_l2(self, images):
        return convolutions(images, self.pca_l2.components_,
                            self.filter_shape_l2,
                            self.step_shape_l2)

    def histogram(self, binary_image):
        """
        Separate a given image into blocks and calculate a histogram
        in each block.

        Supporse data in a block is in range [0, 3] and the acutual
        values are

        ::

            [0 0 1]
            [2 2 2]
            [2 3 3]

        If default bins ``[-0.5 0.5 1.5 2.5 3.5]`` applied,
        then the histogram will be ``[2 1 4 2]``.
        If ``n_bins`` is specified, the range of data divided equally.
        For example, if the data is in range ``[0, 3]`` and ``n_bins = 2``,
        bins will be ``[-0.5 1.5 3.5]`` and the histogram will be ``[3 6]``.
        """

        k = pow(2, self.n_l2_output)
        if self.n_bins is None:
            self.n_bins = k + 1
        bins = np.linspace(-0.5, k - 0.5, self.n_bins)

        patches = Patches(binary_image, self.block_shape, self.block_shape)

        hist = []
        for patch in patches.patches:
            h, _ = np.histogram(patch, bins)
            hist.append(h)
        return np.concatenate(hist)

    def fit(self, images):
        assert(np.ndim(images) == 3)  # input image must be grayscale
        n_images = images.shape[0]
        patches = images_to_patches(images,
                                    self.filter_shape_l1,
                                    self.step_shape_l1)
        self.pca_l1.fit(patches)

        # images.shape == (L1, n_images, y, x)
        images = self.convolution_l1(images)

        # np.vstack(images).shape == (L1 * n_images, y, x)
        patches = images_to_patches(np.vstack(images),
                                    self.filter_shape_l2,
                                    self.step_shape_l2)
        self.pca_l2.fit(patches)
        return self

    def transform(self, images):
        assert(np.ndim(images) == 3)  # input image must be grayscale

        n_images = images.shape[0]

        # images.shape == (n_images, y, x)
        L1 = self.convolution_l1(images)

        # L1.shape == (L1, n_images, y, x)
        # iterate over each L1 output
        X = []
        for maps in L1:
            # maps.shape == (n_images, y, x)
            maps = self.convolution_l2(maps)
            # feature maps are generated.
            # maps.shape == (L2, n_images, y, x) right here
            maps = np.swapaxes(maps, 0, 1)
            # maps.shape == (n_images, L2, y, x)
            x = [self.histogram(binarize(m)) for m in maps]
            X.append(x)
        X = np.array(X)
        # transform X into the form (n_images, n_features).
        # now X.shape == (L1, n_images, n) where n is a value derived
        # from block_shape and the dimensions of each histogram.
        X = np.swapaxes(X, 0, 1)  # X.shape == (n_images, L1, n)
        X = X.reshape(n_images, -1)  # flatten each subarray
        return X.astype(np.float64)  # now X.shape == (n_images, L1*n)

    def structure_is_valid(self, image_shape):
        """
        Check that the filter visits all pixels of input images without
        dropping any information.
        """
        def is_valid_(image_shape, filter_shape, step_shape):
            ys, xs = steps(image_shape, filter_shape, step_shape)
            fh, fw = filter_shape
            h, w = image_shape
            assert(ys[-1]+fh == h)
            assert(xs[-1]+fw == w)
            return output_shape(ys, xs)

        image_shape = to_tuple_if_int(image_shape)

        try:
            output_shape_l1 = is_valid_(image_shape,
                                        self.filter_shape_l1,
                                        self.step_shape_l1)
            output_shape_l2 = is_valid_(output_shape_l1,
                                        self.filter_shape_l2,
                                        self.step_shape_l2)
            is_valid_(output_shape_l2, self.block_shape, self.block_shape)
        except AssertionError:
            return False
        return True
