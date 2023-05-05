import math
import faiss
import numpy as np
from scipy.stats import qmc
from typing import Tuple
from pykdtree.kdtree import KDTree
from sklearn.preprocessing import normalize
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from .utils import sqrt_of_summed_variance


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

def get_outlier_indices(original_data: np.ndarray, outlier_percentile: float = 0.99):
    indices = np.arange(0, original_data.shape[0])
    distances = get_nearest_neighbor_distances(original_data)
    cutoff = np.quantile(distances, outlier_percentile)
    outlier_indices = indices[distances >= cutoff]
    return outlier_indices

def get_local_outlier_factor_indices(original_data: np.ndarray, k_neighbors: int = 20, outlier_score: float = 1.5):
    lof = LocalOutlierFactor(n_neighbors=k_neighbors)
    lof.fit(original_data)
    scores = -lof.negative_outlier_factor_
    all_indices = np.arange(original_data.shape[0])
    return all_indices[scores >= outlier_score]
    
def calculate_outlier_retention(outlier_indices: np.ndarray, subsampled_indices: np.ndarray) -> float:
    num_retained = np.sum(np.isin(outlier_indices, subsampled_indices))
    return num_retained / np.shape(outlier_indices)[0]

def calculate_discrepancy(data):
    # TODO - move normalization to unit hypercube to its own function
    min_vals = data.min(axis=tuple(range(data.ndim - 1)))
    max_vals = data.max(axis=tuple(range(data.ndim - 1)))
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return qmc.discrepancy(normalized_data, workers=-1)
