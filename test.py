import unittest
import numpy as np

from pcanet import Patches, PCANet, image_to_patch_vectors
from pcanet import binarize, binary_to_decimal, to_tuple_if_int
from ensemble import most_frequent_label


try:
    import cupy as xp
    from cupy.testing import assert_array_equal
except ImportError:
    import numpy as xp
    from numpy.testing import assert_array_equal


class TestPatches(unittest.TestCase):
    def test_kernel_startpoints(self):
        """
        The coodinates of startpoints of kernel are valid.
        """
        image_shape = (10, 8)
        filter_shape = (4, 3)
        step_shape = (1, 2)
        image = np.zeros(image_shape)
        patches = Patches(image, filter_shape, step_shape)
        self.assertEqual(list(patches.ys), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(list(patches.xs), [0, 2, 4])

    def test_patches(self):
        # Supporse that image below is geven.
        # [[0 1 2]
        #  [3 4 5]
        #  [6 7 8]]
        #
        # If the patches are squares and its size = 2, and the step size = 1
        # the extracted patches should be like below.
        # [0 1]  [1 2]  [3 4]  [4 5]
        # [3 4]  [4 5]  [6 7]  [7 8]

        image = np.array(
            [[0, 3, 1],
             [3, 1, 1],
             [2, 0, 0]]
        )

        patches = Patches(image, (2, 2), (1, 1)).patches
        expected = np.array([
            [[0, 3],
             [3, 1]],
            [[3, 1],
             [1, 1]],
            [[3, 1],
             [2, 0]],
            [[1, 1],
             [0, 0]]
        ])
        assert_array_equal(patches, expected)


class TestPCANet(unittest.TestCase):
    def test_binarize(self):
        image = np.array([
            [3, -8],
            [2, 1],
            [-1, 5]
        ])
        expected = np.array([
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        assert_array_equal(binarize(image), expected)

    def test_binary_to_decimal(self):
        image = xp.array([
            [[[1, 0],
              [1, 0]],
             [[1, 1],
              [0, 1]]],
            [[[1, 1],
              [0, 0]],
             [[1, 0],
              [1, 0]]]
        ])
        expected = xp.array([
            [[3, 1],
             [2, 1]],
            [[3, 2],
             [1, 0]]
        ])
        assert_array_equal(binary_to_decimal(image), expected)

    def test_histogram(self):
        images = xp.array([
            [[0, 1, 1, 3],
             [3, 1, 2, 2],
             [2, 0, 1, 2],
             [0, 1, 1, 1]],
            [[2, 0, 1, 2],
             [1, 3, 0, 1],
             [2, 2, 2, 3],
             [1, 3, 3, 1]]
        ])
        expected = xp.array([
            [1, 2, 0, 1, 0, 1, 2, 1, 2, 1, 1, 0, 0, 3, 1, 0],
            [1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 2, 1, 0, 1, 1, 2]
        ])
        pcanet = PCANet(None, None, None, None, None, None,
                        n_l2_output=2,
                        filter_shape_pooling=2,
                        step_shape_pooling=2)
        assert_array_equal(pcanet.histogram(images), expected)

        images = xp.array([
            [[1, 0, 1],
             [2, 0, 0],
             [1, 3, 3]],
            [[2, 0, 0],
             [1, 1, 1],
             [3, 0, 1]]
        ])
        expected = xp.array([
            [2, 1, 1, 0, 3, 1, 0, 0, 1, 1, 1, 1, 2, 0, 0, 2],
            [1, 2, 1, 0, 2, 2, 0, 0, 1, 2, 0, 1, 1, 3, 0, 0]
        ])
        pcanet = PCANet(None, None, None, None, None, None,
                        n_l2_output=2,
                        filter_shape_pooling=2,
                        step_shape_pooling=1)
        assert_array_equal(pcanet.histogram(images), expected)

    def test_to_tuple_if_int(self):
        # duplicate if int is given
        self.assertEqual(to_tuple_if_int(10), (10, 10))
        # do nothing if non-integer is given
        self.assertEqual(to_tuple_if_int((10, 10)), (10, 10))

    def test_image_to_patch_vectors(self):
        image = np.array([
            [0, 2, 1, 5],
            [2, 0, 1, 1],
            [3, 3, 0, 2],
        ])
        expected = np.array([
            [-1, 1, 1, -1],
            [1, 0, -1, 0],
            [-1, 3, -1, -1],
            [0, -2, 1, 1],
            [-1, 0, 2, -1],
            [0, 0, -1, 1]
        ])
        patches = image_to_patch_vectors(image, (2, 2), (1, 1))
        assert_array_equal(patches, expected)

    def test_validate_structure(self):
        # Check whether filters visit all pixels of input images
        pcanet = PCANet(
            image_shape=9,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=1, n_l2_output=1,
            filter_shape_pooling=1, step_shape_pooling=1
        )
        pcanet.validate_structure()

        pcanet = PCANet(
            image_shape=10,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=1, n_l2_output=1,
            filter_shape_pooling=1, step_shape_pooling=1
        )
        self.assertRaises(ValueError, pcanet.validate_structure)

        # Check whether filters visit all pixels of L1 output
        # the shape of L1 output is (6, 6)
        pcanet = PCANet(
            image_shape=13,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=1, n_l2_output=1,
            filter_shape_pooling=1, step_shape_pooling=1
        )
        pcanet.validate_structure()

        pcanet = PCANet(
            image_shape=13,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=2, n_l2_output=1,
            filter_shape_pooling=1, step_shape_pooling=1
        )
        self.assertRaises(ValueError, pcanet.validate_structure)

        # Check whether blocks cover all pixels of L2 output
        # the shape of L1 output is (9, 9)
        # the shape of L2 output is (4, 4)
        pcanet = PCANet(
            image_shape=19,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=2, n_l2_output=1,
            filter_shape_pooling=2, step_shape_pooling=2
        )
        pcanet.validate_structure()

        pcanet = PCANet(
            image_shape=19,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=2, n_l2_output=1,
            filter_shape_pooling=3, step_shape_pooling=1
        )
        self.assertRaises(ValueError, pcanet.validate_structure)


class TestBagging(unittest.TestCase):
    def test_most_frequent_label(self):
        v = np.array([0, 1, 1, 3, 2, 0, 1])
        self.assertEqual(most_frequent_label(v), 1)

        v = np.array([0, 2, 1, 2, 2, 1, 0])
        self.assertEqual(most_frequent_label(v), 2)


unittest.main()
