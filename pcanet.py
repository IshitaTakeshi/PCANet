# We recommend you to see [the original paper](https://arxiv.org/abs/1404.3606)
# before reading the code below.

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
    def __init__(self, images, filter_shape, step_shape):
        assert(np.ndim(images) == 3)
        self.images = images
        self.filter_shape = filter_shape

        self.ys, self.xs = steps(images.shape[1:3], filter_shape, step_shape)

    @property
    def patches(self):
        """
        Return image patches of shape
        (n_images, n_patches, filter_height, filter_width)
        """
        return np.array([self.patches_(image) for image in self.images])

    def patches_(self, image):
        fh, fw = self.filter_shape
        it = itertools.product(self.ys, self.xs)
        return [image[y:y+fh, x:x+fw] for y, x in it]

    @property
    def output_shape(self):
        return output_shape(self.ys, self.xs)



def normalize_patches(X):
    """
    Normalize patches for each image.

    Parameters
    ----------
    X: np.ndarray
        Set of patches of shape
        (n_images, n_patches, filter_height, filter_width)
    """
    assert(np.ndim(X) == 4)
    return X - X.mean(axis=1, keepdims=True)


def images_to_patch_vectors(images, filter_shape, step_shape):
    """
    Parameters
    ----------
    images: np.array
        Images to extract patch vectors
    filter_shape: tuple of ints
        The shape of a filter
    step_shape: tuple of ints
        Step height/width of a filter

    Returns
    -------
    X: np.array
        A set of normalized and flattened patches

    Each row of X represents a flattened patch.
    The number of columns is N x M where
    N is the number of images and
    M is the number of patches that can be obtained from one image.
    """
    X = Patches(images, filter_shape, step_shape).patches
    X = normalize_patches(X)
    n_images, n_patches, filter_height, filter_width = X.shape
    return X.reshape(n_images * n_patches, filter_height * filter_width)


def convolution(images, filters, filter_shape, step_shape):
    # filters : [n_filters, filter_height, filter_width]
    # images  : [n_images, image_height, image_width]

    patches = Patches(images, filter_shape, step_shape)

    # the shape of patches.patches is
    # (n_images, n_patches, filter_height, filter_width)
    # and the the shape of filters is
    # [n_filters, filter_height, filter_width].
    # Run convolution by calculating products of patches and filters.
    # The shape of convolution output `X` is
    # (n_images, n_patches, n_filters)
    # where n_patches = output_height * output_width.
    X = np.tensordot(patches.patches, filters, axes=([2, 3], [1, 2]))

    # Reshape X into (n_filters, n_images, n_patches)
    X = np.swapaxes(X, 1, 2)
    X = np.swapaxes(X, 0, 1)

    # Here the shape of X is (n_filters, n_images, n_patches)
    # At the last, reshape X into
    # (n_filters, n_images, output_height, output_width)
    return X.reshape(X.shape[0], X.shape[1], *patches.output_shape)


def binarize(X):
    X[X > 0] = 1
    X[X <= 0] = 0
    return X


def binary_to_decimal(X):
    """
    Parameters
    ----------
    X: np.ndarray
        Feature maps
    """
    # This function expects X of shape (n_images, L2, y, x)
    # as an argument.
    # Let's say that X[k] (0 <= k < n_images) can be represented like
    # X[k] = [map_k[0], map_k[1], ..., map_k[L2-1]]
    # where the shape of each map_k is (y, x).
    # Then we calculate
    # a[0] * map_k[0] + a[1] * map_k[1] + ... + a[L2-1] * map_k[L2-1]
    # for each X[k], where a = [2^(L2-1), 2^(L2-2), ..., 2^0]
    # Therefore, the output shape must be (n_images, y, x)
    a = np.arange(X.shape[1])[::-1]
    a = np.power(2, a)
    return np.tensordot(X, a, axes=([1], [0]))


def to_tuple_if_int(value):
    """
    If int is given, duplicate it and return as a 2 element tuple.
    """
    if isinstance(value, int):
        return (value, value)
    return value


class PCANet(object):
    def __init__(self, image_shape,
                 filter_shape_l1, step_shape_l1, n_l1_output,
                 filter_shape_l2, step_shape_l2, n_l2_output,
                 block_shape):
        """
        Parameters
        ----------
        image_shape: int or sequence of ints
            Input image shape.
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

        self.image_shape = to_tuple_if_int(image_shape)

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
        # (n_filters, filter_height*filter_width)
        #   -> (n_filters, filter_height, filter_width)
        filter_ = self.pca_l1.components_.reshape(-1, *self.filter_shape_l1)
        return convolution(images, filter_,
                           self.filter_shape_l1,
                           self.step_shape_l1)

    def convolution_l2(self, images):
        filter_ = self.pca_l2.components_.reshape(-1, *self.filter_shape_l2)
        return convolution(images, filter_,
                           self.filter_shape_l2,
                           self.step_shape_l2)

    def histogram(self, binary_images):
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
        For example, if the data is in range ``[0, 3]`` and
        ``n_bins = 2``, bins will be ``[-0.5 1.5 3.5]`` and
        the histogram will be ``[3 6]``.
        """

        k = pow(2, self.n_l2_output)
        if self.n_bins is None:
            self.n_bins = k + 1
        bins = np.linspace(-0.5, k - 0.5, self.n_bins)

        def histogram(patches):
            # Convert patches extracted from one image
            # into a feature vector.
            h = [np.histogram(patch, bins)[0] for patch in patches]
            return np.concatenate(h)

        patches = Patches(binary_images, self.block_shape, self.block_shape)
        # The shape of patches.patches is
        # (n_images, n_patches, filter_height, filter_width)
        return np.array([histogram(p) for p in patches.patches])

    def fit(self, images):
        assert(np.ndim(images) == 3)  # input image must be grayscale
        assert(images.shape[1:3] == self.image_shape)
        n_images = images.shape[0]
        patches = images_to_patch_vectors(images,
                                          self.filter_shape_l1,
                                          self.step_shape_l1)
        self.pca_l1.fit(patches)

        # images.shape == (L1, n_images, y, x)
        images = self.convolution_l1(images)

        # np.vstack(images).shape == (L1 * n_images, y, x)
        patches = images_to_patch_vectors(np.vstack(images),
                                          self.filter_shape_l2,
                                          self.step_shape_l2)
        self.pca_l2.fit(patches)
        return self

    def transform(self, images):
        assert(np.ndim(images) == 3)  # input image must be grayscale
        assert(images.shape[1:3] == self.image_shape)

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
            maps = binarize(maps)
            maps = binary_to_decimal(maps)
            # maps.shape == (n_images, y, x)
            x = self.histogram(maps)
            # x is a set of feature vectors.
            # The shape of x is (n_images, vector length)
            X.append(x)
        X = np.hstack(X)
        return X.astype(np.float64)

    def validate_structure(self):
        """
        Check that the filter visits all pixels of input images without
        dropping any information.
        Raise ValueError if the network structure does not satisfy the
        above constraint.
        """
        def is_valid_(input_shape, filter_shape, step_shape):
            ys, xs = steps(input_shape, filter_shape, step_shape)
            fh, fw = filter_shape
            h, w = input_shape
            if ys[-1]+fh != h or xs[-1]+fw != w:
                raise ValueError("Invalid network structure.")
            return output_shape(ys, xs)

        output_shape_l1 = is_valid_(self.image_shape,
                                    self.filter_shape_l1,
                                    self.step_shape_l1)
        output_shape_l2 = is_valid_(output_shape_l1,
                                    self.filter_shape_l2,
                                    self.step_shape_l2)
        is_valid_(output_shape_l2, self.block_shape, self.block_shape)
