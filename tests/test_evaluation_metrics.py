import math
import unittest
import numpy as np
from ccsubsample.subsampling import point_diversity_mean_std, point_diversity_histogram
from numpy.testing import assert_allclose

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.subsampled_points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
            [4, 4, 4],
            [6, 6, 6]
        ])
        
    def test_point_diversity_mean_std(self):
        mean, std = point_diversity_mean_std(self.subsampled_points)
        # Distance formula in 3D = math.sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
        first_four_points_nearest = math.sqrt(3 * math.pow(1, 2))
        last_point_nearest = math.sqrt(3 * math.pow(2, 2))
        num_distances = 5
        expected_mean = \
            (4 * first_four_points_nearest + last_point_nearest) / num_distances
        expected_std = math.sqrt((
                4 * math.pow(first_four_points_nearest - expected_mean, 2) +
                math.pow(last_point_nearest - expected_mean, 2)
        ) / num_distances)
        self.assertEqual(expected_mean, mean)
        self.assertEqual(expected_std, std)

    def test_point_diversity_histogram(self):
        histogram = point_diversity_histogram(self.subsampled_points)
        expected_histogram = np.array([0, 0, 4, 0, 0, 0, 0, 0, 1, 0])
        expected_bin_edges = np.array([1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4])
        assert_allclose(expected_histogram, histogram[0])
        assert_allclose(expected_bin_edges, histogram[1])
        
if __name__ == '__main__':
    unittest.main()
