import math
import numpy as np
from typing import Tuple
from pykdtree.kdtree import KDTree
from sklearn.neighbors import KernelDensity


def get_nearest_neighbor_distances(subsampling_result: np.ndarray) -> np.ndarray:
    kd_tree = KDTree(subsampling_result)
    distances, _ = kd_tree.query(subsampling_result, k=2)
    return distances[:, 1]


def point_diversity_mean_std(subsampling_result: np.ndarray) -> Tuple:
    distances = get_nearest_neighbor_distances(subsampling_result)
    mean = np.mean(distances)
    std = np.std(distances)
    return mean, std


def point_diversity_histogram(subsampling_result: np.ndarray, num_bins=10):
    distances = get_nearest_neighbor_distances(subsampling_result)
    floor = math.floor(distances.min())
    ceil = math.ceil(distances.max())
    histogram = np.histogram(distances, bins=num_bins, range=(floor, ceil))
    return histogram


def number_of_points_remaining(subsampling_result: np.ndarray) -> int:
    return subsampling_result.shape[0]



def point_diversity_kde(subsampling_result: np.ndarray) -> np.ndarray:
    distances = get_nearest_neighbor_distances(subsampling_result).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.03).fit(distances)
    x_axis_points = np.linspace(0, np.max(distances), np.size(distances)).reshape(-1, 1)
    log_dens = kde.score_samples(x_axis_points)
    return x_axis_points, log_dens
    