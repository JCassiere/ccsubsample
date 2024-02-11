import unittest
import numpy as np
from ccsubsample.subsampling.subsampling import faiss_ivf_subsample

class MyTestCase(unittest.TestCase):
    def dataset_with_duplicates(self, num_unique_points, num_duplicates):
        return np.concatenate([np.tile(np.repeat(i, 5), (num_duplicates, 1)) for i in range(num_unique_points)])
    # def setUp(self) -> None:
    #     self.subsampled_points = np.array([
    #         [0, 0, 0],
    #         [1, 1, 1],
    #         [3, 3, 3],
    #         [4, 4, 4],
    #         [6, 6, 6]
    #     ])
        
    def test_all_points_have_true_duplicates(self):
        dataset = self.dataset_with_duplicates(100, 100)
        indices = faiss_ivf_subsample(dataset)
        self.assertGreater(dataset.shape[0], len(indices))

    def test_all_points_are_the_same_duplicate(self):
        dataset = self.dataset_with_duplicates(1, 1000)
        indices = faiss_ivf_subsample(dataset)
        result = dataset[indices]
        np.testing.assert_array_equal(result, dataset[0:1, :])