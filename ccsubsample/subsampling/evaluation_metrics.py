import math
import faiss
import numpy as np
from typing import Tuple
from pykdtree.kdtree import KDTree
from sklearn.preprocessing import normalize
from sklearn.neighbors import KernelDensity


def get_nearest_neighbor_distances(subsampling_result: np.ndarray) -> np.ndarray:
    _, dim = subsampling_result.shape
    index = faiss.IndexFlatL2(dim)
    index.add(subsampling_result)
    d, i = index.search(subsampling_result, 2)
    self_indices = np.arange(0, subsampling_result.shape[0])
    distances = d[:, 1]
    distances[i[:, 0] != self_indices] = d[:, 0][i[:, 0] != self_indices]
    return distances


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


def point_diversity_kde(subsampling_result: np.ndarray, bandwidth=0.5) -> np.ndarray:
    distances = get_nearest_neighbor_distances(subsampling_result).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(distances)
    x_axis_points = np.linspace(0, np.max(distances), np.size(distances)).reshape(-1, 1)
    log_dens = kde.score_samples(x_axis_points)
    return x_axis_points, log_dens
    