import unittest
import numpy as np
from numpy.testing import assert_array_equal

from pcanet import Patches, PCANet, PatchNormalizer, convolution, binarize
from pcanet import to_tuple_if_int


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

        images = np.array([
            [[1, 3, 2],
             [4, 1, 5],
             [3, 2, 6]]
        ])
        filter_shape = (2, 2)
        filter_ = np.array([
            [1, 2],
            [3, 1]
        ])
        T = convolution(images, filter_, filter_shape, (1, 1))
        expected = np.array([[
            [20, 15],
            [17, 23]
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

        pcanet = PCANet(None, None, None, None, None, None,
                        n_l2_output=2, block_shape=(3, 2))
        # assume that n_l1_output = 2
        assert_array_equal(pcanet.histogram(image), expected)

    def test_to_tuple_if_int(self):
        # duplicate if int is given
        self.assertEqual(to_tuple_if_int(10), (10, 10))
        # do nothing if non-integer is given
        self.assertEqual(to_tuple_if_int((10, 10)), (10, 10))

    def test_patch_normalizer(self):
        patches = np.array([
            [[0, 2],
             [2, 1]],
            [[1, 3],
             [3, 0]],
            [[2, 1],
             [1, 2]]
        ])

        normalizer = PatchNormalizer(patches)

        expected = np.array([
            [[1, 2],
             [2, 1]]
        ])
        assert_array_equal(normalizer.mean, expected)

        # normalize a patch by subtracting `normalizer.mean`
        patch = np.array([
            [1, 3],
            [2, 1]
        ])
        expected = np.array([
            [0, 1],
            [0, 0]
        ])
        assert_array_equal(normalizer.normalize_patch(patch), expected)

        # normalize multiple patches
        patches = np.array([
            [[0, 3],
             [1, 0]],
            [[1, 1],
             [2, 3]]
        ])
        expected = np.array([
            [[-1, 1],
             [-1, -1]],
            [[0, -1],
             [0, 2]]
        ])
        assert_array_equal(normalizer.normalize_patches(patches), expected)

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
