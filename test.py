import unittest
import numpy as np
from numpy.testing import assert_array_equal

from pcanet import Patches, PCANet, normalize_patches, convolution, binarize
from pcanet import binary_to_decimal, to_tuple_if_int


class TestPatches(unittest.TestCase):
    def test_kernel_startpoints(self):
        """
        The coodinates of startpoints of kernel are valid.
        """
        image_shape = (10, 8)
        filter_shape = (4, 3)
        step_shape = (1, 2)
        images = np.zeros((1, *image_shape))
        patches = Patches(images, filter_shape, step_shape)
        self.assertEqual(list(patches.ys), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(list(patches.xs), [0, 2, 4])

    def test_patches(self):
        # Supporse that images below aregeven.
        # [[0 1 2]
        #  [3 4 5]
        #  [6 7 8]]
        # [[0 3 1]
        #  [3 1 1]
        #  [2 0 0]]
        #
        # If the patches are squares and its size = 2, and the step size = 1
        # then the extracted patches should be like below.
        # From the first image:
        # [0 1]  [1 2]  [3 4]  [4 5]
        # [3 4]  [4 5]  [6 7]  [7 8]
        # From the second image:
        # [0 3]  [3 1]  [3 1]  [1 1]
        # [3 1]  [1 1]  [2 0]  [0 0]
        images = np.array([
         [[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]],
         [[0, 3, 1],
          [3, 1, 1],
          [2, 0, 0]]
        ])

        patches = Patches(images, (2, 2), (1, 1)).patches
        expected = np.array([
            [[[0, 1],
              [3, 4]],
             [[1, 2],
              [4, 5]],
             [[3, 4],
              [6, 7]],
             [[4, 5],
              [7, 8]]],
            [[[0, 3],
              [3, 1]],
             [[3, 1],
              [1, 1]],
             [[3, 1],
              [2, 0]],
             [[1, 1],
              [0, 0]]],
        ])
        assert_array_equal(patches, expected)


class TestPCANet(unittest.TestCase):
    def test_images_to_patches(self):
        # TODO
        pass

    def test_convolution(self):
        images = np.array([
            [[0, 2, 1],
             [2, 3, 0],
             [1, 1, 1]],
            [[3, 1, 0],
             [1, 2, 2],
             [2, 1, 1]]
        ])
        filters = np.array([
            [[1, 1],
             [1, 1]],
            [[1, 1],
             [0, 2]]
        ])
        T = convolution(images, filters, (2, 2), (1, 1))
        expected = np.array([
            [[[7, 6],
              [7, 5]],
             [[7, 5],
              [6, 6]]],
            [[[8, 3],
              [7, 5]],
             [[8, 5],
              [5, 6]]]
        ])
        assert_array_equal(T, expected)

        # images = np.array([
        #     [[1, 3, 2],
        #      [4, 1, 5],
        #      [3, 2, 6]]
        # ])
        # filter_ = np.array([
        #     [1, 2],
        #     [3, 1]
        # ])
        # T = convolution(images, filter_, (2, 2), (1, 1))
        # expected = np.array([[
        #     [20, 15],
        #     [17, 23]
        # ]])
        # assert_array_equal(T, expected)

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
        image = np.array([
            [[[1, 0],
              [1, 0]],
             [[1, 1],
              [0, 1]]],
            [[[1, 1],
              [0, 0]],
             [[1, 0],
              [1, 0]]]
        ])
        expected = np.array([
            [[3, 1],
             [2, 1]],
            [[3, 2],
             [1, 0]]
        ])
        assert_array_equal(binary_to_decimal(image), expected)

    def test_histogram(self):
        images = np.array([
            [[0, 1, 1, 3],
             [3, 1, 2, 2],
             [2, 0, 1, 2],
             [0, 1, 1, 1]],
            [[2, 0, 1, 2],
             [1, 3, 0, 1],
             [2, 2, 2, 3],
             [1, 3, 3, 1]]
        ])
        expected = np.array([
            [1, 2, 0, 1, 0, 1, 2, 1, 2, 1, 1, 0, 0, 3, 1, 0],
            [1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 2, 1, 0, 1, 1, 2]
        ])
        pcanet = PCANet(None, None, None, None, None, None,
                        n_l2_output=2, block_shape=(2, 2))
        # assume that n_l1_output = 2
        assert_array_equal(pcanet.histogram(images), expected)

    def test_to_tuple_if_int(self):
        # duplicate if int is given
        self.assertEqual(to_tuple_if_int(10), (10, 10))
        # do nothing if non-integer is given
        self.assertEqual(to_tuple_if_int((10, 10)), (10, 10))

    def test_normalize_patches(self):
        patches = np.array([
            [[[0, 3],
              [1, 5]],
             [[2, 1],
              [1, 1]]],
            [[[1, 3],
              [0, 2]],
             [[1, 1],
              [2, 2]]]
        ])
        expected = np.array([
            [[[-1, 1],
              [0, 2]],
             [[1, -1],
              [0, -2]]],
            [[[0, 1],
              [-1, 0]],
             [[0, -1],
              [1, 0]]]
        ])
        assert_array_equal(normalize_patches(patches), expected)

    def test_validate_structure(self):
        # Check whether filters visit all pixels of input images
        pcanet = PCANet(
            image_shape=9,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=1, n_l2_output=1,
            block_shape=1
        )
        pcanet.validate_structure()

        pcanet = PCANet(
            image_shape=10,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=1, n_l2_output=1,
            block_shape=1
        )
        self.assertRaises(ValueError, pcanet.validate_structure)

        # Check whether filters visit all pixels of L1 output
        # the shape of L1 output is (6, 6)
        pcanet = PCANet(
            image_shape=13,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=1, n_l2_output=1,
            block_shape=1
        )
        pcanet.validate_structure()

        pcanet = PCANet(
            image_shape=13,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=2, n_l2_output=1,
            block_shape=1
        )
        self.assertRaises(ValueError, pcanet.validate_structure)

        # Check whether blocks cover all pixels of L2 output
        # the shape of L1 output is (9, 9)
        # the shape of L2 output is (4, 4)
        pcanet = PCANet(
            image_shape=19,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=2, n_l2_output=1,
            block_shape=2
        )
        pcanet.validate_structure()

        pcanet = PCANet(
            image_shape=19,
            filter_shape_l1=3, step_shape_l1=2, n_l1_output=1,
            filter_shape_l2=3, step_shape_l2=2, n_l2_output=1,
            block_shape=3
        )
        self.assertRaises(ValueError, pcanet.validate_structure)

unittest.main()
