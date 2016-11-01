# We recommend you to see [the original paper](https://arxiv.org/abs/1404.3606)
# before reading the code below.

# TODO use sphinx
import itertools

from scipy import linalg
import numpy as np
from sklearn.decomposition import PCA


class Patches(object):
    def __init__(self, image, filter_shape, step_shape):
        self.image = image

        self.filter_shape = filter_shape
        self.step_shape = step_shape

        h, w = image.shape
        fh, fw = filter_shape
        sh, sw = step_shape

        self.ys = range(0, h-fh+1, sh)
        self.xs = range(0, w-fw+1, sw)

    def patches_with_indices(self):
        """
        Yields patches with its location indices.
        Kernel visits the image from the left top into right bottom.
        :j:
        :i:
        :patch: (j, i)th patch
        """

        # The behaviour is as same as below:
        # ```
        # for j, y in enumerate(self.ys):
        #     for i, x in enumerate(self.xs):
        #         yield j, i, self.image[y:y+sh, x:x+sw]
        # ```
        # But the code above does not work if called multiple times,
        # so we create a generator object every time of function call.
        fh, fw = self.filter_shape
        it = itertools.product(enumerate(self.ys), enumerate(self.xs))
        return ((j, i, self.image[y:y+fh, x:x+fw]) for (j, y), (i, x) in it)

    def patches(self):
        """
        Return patches
        """
        return np.array([p for j, i, p in self.patches_with_indices()])

    @property
    def output_shape(self):
        return len(self.ys), len(self.xs)


def heaviside_step(X):
    X[X>0] = 1
    X[X<=0] = 0
    return X


def normalize(X):
    """Subtract mean so that the mean of X be a zero vector"""
    return X - X.mean(axis=0, keepdims=True)


def images_to_patches(images, filter_shape, step_shape):
    """
    Each row of X represents a flattened patch
    The number of columns is (number of images) x (number of
    patches that can be obtained from one image).
    """
    def f(image):
        X = Patches(image, filter_shape, step_shape).patches()
        X = X.reshape(X.shape[0], -1)  # reshape each patch into a vector
        return normalize(X)

    return np.vstack([f(image) for image in images])


def convolution(images, filter_, filter_shape, step_shape):
    def convolution_(patches):
        L = np.empty(patches.output_shape)
        for j, i, patch in patches.patches_with_indices():
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
    for i, image in enumerate(reversed(images)):  # TODO check the order
        output += np.power(2, i) * heaviside_step(image)
    return output


class PCANet(object):
    def __init__(self,
                 filter_shape_l1, step_shape_l1, n_l1_output,
                 filter_shape_l2, step_shape_l2, n_l2_output,
                 block_shape):
        """
        :n_l1_output: L1 in the original paper. The number of outputs
        obtained from a set of input images.
        :n_l2_output: L2 in the original paper. The number of outputs
        obtained from each L1 output.
        """
        self.filter_shape_l1 = filter_shape_l1
        self.step_shape_l1 = step_shape_l1
        self.n_l1_output = n_l1_output

        self.filter_shape_l2 = filter_shape_l2
        self.step_shape_l2 = step_shape_l2
        self.n_l2_output = n_l2_output

        self.block_shape = block_shape
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
        # Separate a given image into blocks and calculate a histogram
        # in each block
        #
        # [       ]            [    ]|[    ]
        # [       ]            [    ]|[    ]
        # [       ]  ------>   ------+------
        # [       ]            [    ]|[    ]
        #                      [    ]|[    ]
        #
        # Supporse data in a block is in range [0, 3] and the acutual
        # values is
        # [0 0 1]
        # [2 2 2]
        # [2 3 3],
        # then the default bins will be [-0.5 0.5 1.5 2.5 3.5]
        # and the histogram will be [2 1 4 2].
        # The range of data divided equally if n_bins is specified.
        # For example, if the data is in range [0, 3] and n_bins = 2,
        # bins will be [-0.5 1.5 3.5] and then the histogram will be [3 6].

        k = pow(2, self.n_l2_output)
        if self.n_bins is None:
            self.n_bins = k + 1
        bins = np.linspace(-0.5, k - 0.5, self.n_bins)

        patches = Patches(binary_image, self.block_shape, self.block_shape)

        hist = []
        for patch in patches.patches():
            h, _ = np.histogram(patch, bins)
            hist.append(h)
        return np.concatenate(hist)

    def fit(self, images):
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
        print("n_images: " + str(n_images))

        # images.shape == (n_images, y, x)
        L1 = self.convolution_l1(images)
        # now images.shape == (L1, n_images, y, x)

        X = []
        for T in L1:
            # input T of shape (n_images, y, x)
            T = self.convolution_l2(T)  # T.shape == (L2, n_images, y, x)
            T = np.swapaxes(T, 0, 1)  # T.shape == (n_images, L2, y, x)
            x = [self.histogram(binarize(t)) for t in T]
            X.append(x)
        X = np.array(X)
        # TODO explain
        X = np.swapaxes(X, 0, 1)
        return X.reshape(n_images, -1)
