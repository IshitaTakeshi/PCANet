import unittest
import numpy as np
from numpy.testing import assert_array_equal

from pcanet import Patches, PCANet, convolution, normalize, binarize


class TestPatches(unittest.TestCase):
    def test_kernel_startpoints(self):
        """
        The coodinates of startpoints of kernel are valid.
        """
        image_shape = (10, 8)
        filter_shape = (4, 3)
        step_shape = (1, 2)
        patches = Patches(np.zeros(image_shape), filter_shape, step_shape)
        self.assertEqual(list(patches.ys), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(list(patches.xs), [0, 2, 4])

    def test_patches_with_indices(self):
        # Supporse that an image below is geven.
        # [0 1 2]
        # [3 4 5]
        # [6 7 8]
        #
        # If the patches are squares and its size = 2 and the step size = 1,
        # then the extracted patches should be like below.
        #
        # (0, 0)th patch:
        # [0 1]
        # [3 4]
        #
        # (0, 1)th patch:
        # [1 2]
        # [4 5]
        #
        # (1, 0)th patch:
        # [3 4]
        # [6 7]
        #
        # (1, 1)th patch:
        # [4 5]
        # [7 8]

        image = np.arange(9).reshape(3, 3)
        patches = Patches(image, (2, 2), (1, 1))
        expected_coordinates = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected_patches = np.array([
            [[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]],
            [[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]
        ])
        for index, (j, i, patch) in enumerate(patches.patches_with_indices):
            self.assertEqual([j, i], expected_coordinates[index])
            assert_array_equal(patch, expected_patches[index])

    def test_patches(self):
        # Supporse that an image below is geven.
        # [0 1 2]
        # [3 4 5]
        # [6 7 8]
        # If the patches are squares and its size = 2, and the step size = 1
        # then the extracted patches should be like below.
        # [0 1]  [1 2]  [3 4]  [4 5]
        # [3 4]  [4 5]  [6 7]  [7 8]
        image = np.arange(9).reshape(3, 3)
        patches = Patches(image, (2, 2), (1, 1)).patches
        expected = np.array([
            [[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]],
            [[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]
        ])
        assert_array_equal(patches, expected)


class TestPCANet(unittest.TestCase):
    def test_normalize(X):
        X = np.array([
            [0, 1, 3, -3, 4, 2],
            [8, 2, 3, 1, -6, 5]
        ])
        assert_array_equal(np.mean(normalize(X), axis=0), np.zeros(6))

    def test_images_to_patches(self):
        # TODO
        pass

    def test_convolution(self):
        images = np.arange(9).reshape(1, 3, 3)
        filter_shape = (2, 2)
        filter_ = np.ones(filter_shape)
        T = convolution(images, filter_, filter_shape, (1, 1))
        expected = np.array([[
            [8, 12],
            [20, 24]
        ]])
        assert_array_equal(T, expected)

    def test_binarize(self):
        image = np.array([
            [[1, 1],
             [0, 0]],
            [[1, 0],
             [1, 0]]
        ])
        expected = np.array([
            [3, 2],
            [1, 0]
        ])
        assert_array_equal(expected, binarize(image))

    def test_histogram(self):
        image = np.array([
            [0, 1, 1, 3],
            [3, 1, 2, 2],
            [2, 0, 1, 2],
            [2, 0, 1, 2],
            [1, 3, 0, 1],
            [2, 2, 2, 3]
        ])
        expected = np.array([
            2, 2, 1, 1,
            0, 2, 3, 1,
            1, 1, 3, 1,
            1, 2, 2, 1
        ])

        pcanet = PCANet(None, None, None, None, None,
                        n_l2_output=2, block_shape=(3, 2))
        # assume that n_l1_output = 2
        assert_array_equal(pcanet.histogram(image), expected)

    def test_pca(self):
        pass


unittest.main()
