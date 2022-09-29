import math
import numpy as np
from pykdtree.kdtree import KDTree


def get_nearest_neighbor_distances(subsampling_result):
    kd_tree = KDTree(subsampling_result)
    distances, _ = kd_tree.query(subsampling_result, k=2)
    return distances[:, 1]


def point_diversity_mean_std(subsampling_result):
    distances = get_nearest_neighbor_distances(subsampling_result)
    mean = np.mean(distances)
    std = np.std(distances)
    return mean, std


def point_diversity_histogram(subsampling_result, num_bins=10):
    distances = get_nearest_neighbor_distances(subsampling_result)
    floor = math.floor(distances.min())
    ceil = math.ceil(distances.max())
    histogram = np.histogram(distances, bins=num_bins, range=(floor, ceil))
    return histogram
