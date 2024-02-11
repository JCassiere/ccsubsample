from __future__ import print_function

import math
import random
import time
import numpy as np
import multiprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykdtree.kdtree import KDTree
from .utils import reduce_dimensions_with_pca, scale_and_standardize_data, get_image_indices_to_keep
import faiss
from scipy.stats import qmc
import torch
from torch_geometric.nn.pool import fps


def pykdtree_query(full_data, data_slice):
    kd_tree = KDTree(full_data, leafsize=6)
    return kd_tree.query(data_slice, k=2)

    
def single_process_kdtree_query(remaining_datapoints):
    kd_tree = KDTree(remaining_datapoints, leafsize=6)
    # get 2 nearest neighbors, because the first closest will be the point itself
    distances, indices = kd_tree.query(remaining_datapoints, k=2)
    self_indices = np.arange(0, remaining_datapoints.shape[0])
    test = np.sum(self_indices != indices[:, 0])
    return distances, indices


def flat_index(remaining_datapoints):
    n, dim = remaining_datapoints.shape
    index = faiss.IndexFlatL2(dim)
    index.add(remaining_datapoints)
    return index


def ivf_index(remaining_datapoints):
    n, dim = remaining_datapoints.shape
    if n < 10000:
        return flat_index(remaining_datapoints)
    quantizer = faiss.IndexFlatL2(dim)
    nlists = 2 ** math.floor(math.log(n / 10))
    index = faiss.IndexIVFFlat(quantizer, dim, nlists)
    t = time.time()
    index.train(remaining_datapoints)
    index.nprobe = 8
    print("training time: {}".format(time.time() - t))
    index.add(remaining_datapoints)
    return index


def ivfpq_index(remaining_datapoints):
    # TODO - automatically perform dimensionality reduction so dim % m == 0
    n, dim = remaining_datapoints.shape
    if n < 10000:
        return flat_index(remaining_datapoints)
    m = 8
    nbits = min(math.floor(math.log(n / 10)), 8)
    quantizer = faiss.IndexFlatL2(dim)
    nlists = 2 ** math.floor(math.log(n / 10))
    index = faiss.IndexIVFPQ(quantizer, dim, nlists, m, nbits)
    t = time.time()
    index.train(remaining_datapoints)
    index.nprobe = 8
    print("training time: {}".format(time.time() - t))
    index.add(remaining_datapoints)
    return index

    
def hnsw_index(remaining_datapoints):
    n, dim = remaining_datapoints.shape
    if n < 10000:
        return flat_index(remaining_datapoints)
    m = 32
    nbits = min(math.floor(math.log(n / 10)), 8)
    index = faiss.IndexHNSWFlat(dim, m)
    t = time.time()
    print("training time: {}".format(time.time() - t))
    index.add(remaining_datapoints)
    return index


def faiss_query(remaining_datapoints, index):
    # TODO - automatically perform dimensionality reduction so dim % m == 0
    k = 2
    t = time.time()
    d, i = index.search(remaining_datapoints, k)
    print("search time: {}".format(time.time() - t))
    self_indices = np.arange(0, remaining_datapoints.shape[0])
    distances = d[:, 1]
    indices = i[:, 1]
    # faiss does not find a vector to be its own nearest neighbor 100% of the time,
    # so we need to make sure we keep the vector's 1st nearest neighbor in those cases
    distances[i[:, 0] != self_indices] = d[:, 0][i[:, 0] != self_indices]
    indices[i[:, 0] != self_indices] = i[:, 0][i[:, 0] != self_indices]
    distances = np.concatenate((np.zeros((distances.shape[0], 1)), distances[:, None]), axis=1)
    indices = np.concatenate((self_indices[:, None], indices[:, None]), axis=1)
    
    return distances, indices


def multi_process_kdtree_query(remaining_datapoints, num_cpus_to_not_use=1):
    cpus = multiprocessing.cpu_count() - num_cpus_to_not_use
    pool = multiprocessing.get_context("forkserver").Pool(processes=cpus)
    data_len = len(remaining_datapoints)
    slice_size = data_len // cpus
    slices = []
    for i in range(cpus):
        if i == cpus - 1:
            data_slice = remaining_datapoints[i * slice_size:]
        else:
            data_slice = remaining_datapoints[i * slice_size:(i + 1) * slice_size]
        slices.append(data_slice)
    results_async = [pool.apply_async(pykdtree_query, args=(remaining_datapoints, slices[i])) for i in range(cpus)]
    results = [res.get() for res in results_async]
    distances, indices = results[0]
    for i in range(1, cpus):
        distances = np.append(distances, results[i][0], 0)
        indices = np.append(indices, results[i][1], 0)
    pool.close()
    return distances, indices

def find_keep_remove_pairs(removal_candidate_indices_with_neighbor):
    """
    Find pairs of nearest neighbors where one of the points should be removed
    
    :param removal_candidate_indices_with_neighbor:
    :return: keep_remove_pairs: List[(int, int)] - a list of pairs of indices, where the first
        in each pair represents the index of a nearest neighbor to keep, and the second represents
        the index of a nearest neighbor to remove
    """
    # Go through the list of removal candidates paired with their neighbors
    # Mark each neighbor for removal. While going through the list, do
    # not check any point that has already been marked for removal
    # (i.e. any point that was the nearest neighbor for an already-
    # checked point)
    candidates = {}
    keep_remove_pairs = []
    for point, neighbor in removal_candidate_indices_with_neighbor:
        # the nearest neighbor search algorithm is approximate and not exact
        # so you can't count on the algorithm finding both a point and its neighbor
        # to have a nearest neighbor less than the cutoff
        # so mark them both as deletion candidates here
        # otherwise the loop below could error
        candidates[point] = True
        candidates[neighbor] = True
    
    for point, neighbor in removal_candidate_indices_with_neighbor:
        if candidates[point] and candidates[neighbor]:
            keep_remove_pairs.append((point, neighbor))
            candidates[neighbor] = False
    
    return keep_remove_pairs
    
    
def find_keep_indices(removal_candidate_indices_with_neighbor, keep_remove_pairs):
    remove_indices = [remove for (_, remove) in keep_remove_pairs]
    # Don't just keep all the keep indices from keep_remove_pairs
    # instead, remove the remove_indices from the original removal_candidate_indices_with_neighbor array
    keep_indices = np.array(list(set(removal_candidate_indices_with_neighbor[:, 0]) - set(remove_indices)))
    return keep_indices
    
    
def kmeans_cluster_fn(remaining_datapoints, num_clusters):
    _, dim = remaining_datapoints.shape
    kmeans = faiss.Kmeans(dim, num_clusters)
    kmeans.train(remaining_datapoints)
    return kmeans.centroids


def cluster_subsample(data, num_points_desired, cluster_fn):
    remaining_datapoints = np.asarray(data)
    clusters = cluster_fn(remaining_datapoints, num_points_desired)
    _, dim = remaining_datapoints.shape
    all_points_index = faiss.IndexFlatL2(dim)
    all_points_index.add(remaining_datapoints)
    _, selected_indices = all_points_index.search(clusters, 1)
    return selected_indices.reshape(-1)
    
    
def kmeans_subsample(data, num_points_desired):
    return cluster_subsample(data, num_points_desired, kmeans_cluster_fn)


def faiss_subsample(data, index_fn, cutoff_percentile, verbose=1):
    """
    Using Nearest-Neighbor search based algorithm, find the list of indices of the subsampled dataset
    Parameters
    -------------
    :param data: the list of data to subsample
    :param index_fn: function for creating the faiss index
    :param cutoff_percentile: float - a percentile that will be used to calculate the cutoff distance
                                      between data points. Any point whose nearest neighbor is below the
                                      cutoff distance is a candidate for removal from the dataset.
    :param verbose: int - level of verbosity
    Return
    -------------
    overall_keep_list: The list of indices of the final subsampled entries
    """
    start = 0
    
    if verbose >= 1:
        start = time.time()
        print("Started NN-ccsubsample, original length: {}".format(len(data)))
    
    # initialize the index
    original_data_indices = np.arange(len(data))
    permanent_keep_indices = set()
    
    remaining_datapoints = np.asarray(data)
    
    # calculate cutoff as some percentile of nearest neighbor distances
    index = index_fn(remaining_datapoints)
    distances, indices = faiss_query(remaining_datapoints, index)
    distances = np.sqrt(distances)
    cutoff = np.quantile(distances, cutoff_percentile)
    # TODO - if raw cutoff is 0, run subroutine that deletes one member of each pair of nearest neighbors
    #  where the distance between them is 0. Will have to keep doing this until the cutoff is > 0
    #  could also just automatically go through and delete points where distance is 0 first, but that would add
    #  time unnecessarily for data where the cutoff is > 0
    first_loop = True
    keep_going = True
    iter_count = 1
    old_overall_keep_len = 0
    iter_start = 0
    while keep_going:
        if verbose >= 2:
            print("Start iteration {}, Sampleable data points remaining: {}".format(iter_count,
                                                                                    len(original_data_indices)))
            iter_start = time.time()
        
        # reuse distances calculated for cutoff
        if first_loop:
            first_loop = False
        else:
            # build and query nearest neighbour model
            index = index_fn(remaining_datapoints)
            distances, indices = faiss_query(remaining_datapoints, index)
            # Faiss uses the squared distance, so take the square root
            distances = np.sqrt(distances)

        # if distance between a point and its nearest neighbor is below cutoff distance,
        # add the pair's indices (for this iteration) to the candidate removal list
        removal_candidate_indices_with_neighbor = indices[:][distances[:, 1] <= cutoff]
        
        # if distance between a point and its nearest neighbor is above the cutoff distance,
        # the former point can never be removed, so add it to the permanent keep list
        iteration_permanent_keeps = original_data_indices[distances[:, 1] > cutoff]
        
        # set aside any data points above the cutoff, since they can never be removed
        permanent_keep_indices = permanent_keep_indices.union(list(iteration_permanent_keeps))
        
        keep_remove_pairs = find_keep_remove_pairs(removal_candidate_indices_with_neighbor)
        keep_indices = find_keep_indices(removal_candidate_indices_with_neighbor, keep_remove_pairs)
        
        # if keep_indices is empty, slicing like original_data_indices[keep_indices] will give an error
        if len(keep_indices) == 0:
            original_data_indices = np.array([])
            remaining_datapoints = np.array([])
        else:
            original_data_indices = original_data_indices[keep_indices]
            remaining_datapoints = remaining_datapoints[keep_indices]
        
        
        overall_keep_len = original_data_indices.size + len(permanent_keep_indices)
        # remaining_datapoints length can be 0 if all remaining points have been added to the
        # permanent keep list
        # TODO - these two conditions may be logically equivalent
        if overall_keep_len == old_overall_keep_len or len(remaining_datapoints) == 0:
            keep_going = False
        
        if verbose >= 2:
            total_remaining_length = len(original_data_indices) + len(permanent_keep_indices)
            iter_time = time.time() - iter_start
            to_print = "End iteration {}. Total data points remaining: {}\t Time:{}"
            print(to_print.format(iter_count, total_remaining_length, iter_time))
            iter_count += 1
        old_overall_keep_len = overall_keep_len
    if verbose >= 1:
        total_remaining_length = len(original_data_indices) + len(permanent_keep_indices)
        total_time = time.time() - start
        to_print = "End NN-ccsubsample. Data points remaining: {}\t Time:{}"
        print(to_print.format(total_remaining_length, total_time))
    data_indices = sorted(list(original_data_indices) + list(permanent_keep_indices))
    return data_indices

def faiss_flat_subsample(data, cutoff_percentile=0.99, verbose=1):
    return faiss_subsample(data, flat_index, cutoff_percentile, verbose)

def faiss_ivf_subsample(data, cutoff_percentile=0.99, verbose=1):
    return faiss_subsample(data, ivf_index, cutoff_percentile, verbose)

def faiss_ivfpq_subsample(data, cutoff_percentile=0.99, verbose=1):
    return faiss_subsample(data, ivfpq_index, cutoff_percentile, verbose)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def farthest_point_sampling(data, num_points):
    torch_data = torch.from_numpy(data)
    indices = fps(torch_data, batch=None, ratio=num_points / len(data), random_start=True)
    return np.asarray(indices)

def farthest_point_sampling_batched(data, num_points):
    torch_data = torch.from_numpy(data)
    cpu_count = multiprocessing.cpu_count()
    batch_length = torch_data.size()[0] // cpu_count
    offset = torch_data.size()[0] - (batch_length * cpu_count)
    batches = [torch.ones(batch_length, dtype=torch.long) * i for i in range(cpu_count - 1)]
    batches += [torch.ones(batch_length + offset, dtype=torch.long) * (cpu_count - 1)]
    batch = torch.cat(batches)
    indices = fps(torch_data, batch=batch, ratio=num_points / len(data), random_start=True)
    return np.asarray(indices)

def sobol_subsample(data, num_points_desired):
    _, dim = data.shape
    sampler = qmc.Sobol(dim)
    m = math.ceil(math.log(num_points_desired, 2))
    sample = sampler.random_base2(m)
    
    # convert original data to have range [0, 1) in all dimensions
    min_vals = data.min(axis=tuple(range(data.ndim - 1)))
    max_vals = data.max(axis=tuple(range(data.ndim - 1)))
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    
    # create nearest neighbor index of normalized data
    index = ivf_index(normalized_data)
    distances, indices = index.search(sample, k=1)
    # TODO - need to make sure taking only the first num_points_desired_points
    #  is valid
    return indices[:num_points_desired].reshape(-1)
    # return indices.reshape(-1)


def update_clusters(keep_remove_pairs, clusters):
    for keep, remove in keep_remove_pairs:
        clusters[keep] += clusters[remove]
        clusters.pop(remove)
    return clusters


def num_large_clusters(clusters, N=5):
    """
    Count the clusters with at least N members
    :param clusters:
    :return:
    """
    non_singleton_cluster_positions = [1 if len(x) >= N else 0 for x in clusters.values()]
    return sum(non_singleton_cluster_positions)

def num_small_clusters(clusters, N=5):
    singleton_cluster_positions = [1 if len(x) < N else 0 for x in clusters.values()]
    return sum(singleton_cluster_positions)


def batch_subsampling(
    data,
    batch_size=3000000,
    standard_scale=True,
    cutoff_sig=0.2,
    verbose=1,
    shuffle=True,
    data_indices=None
):
    
    """
    Subsample in batches
    This is to save the memory if the data set of interest is too large.
    The data set will first be broken down into equally sized batchess (defined by batch size)
    that will be subsampled individually.
    The resulting subsampled datapoints will then be pooled together for an overall subsample
    If the final pooled set of subsampled datapoints is larger than batch_size,
    batch_subsampling_with_PCA will be called recursively on that set of datapoints
    Parameters
    -------------
    data: List. the original list of data points
    batch_size [1000000]: Int. the number of datapoints in each batch
    standard_scale [True]: Boolean. Whether to apply standard scaler to the dataset prior to ccsubsample
    cutoff_sig [0.02]: Float. cutoff significance. the cutoff distance equals to the Euclidean
                       norm of the standard deviations in all dimensions of the data points
    rate [0.3]: Float. possibility of deletion
    verbose [1]: integer. level of verbosity
    shuffle [True]: Boolean. whether to shuffle the dataset before breaking down into batchs
    Return
    -------------
    sampling_result : the result list of subsampled data points
    """

    if not data_indices:
        data_indices = np.arange(start=0, stop=len(data))
    subsampled_data_indices = []

    if shuffle:
        random.shuffle(data)

    num_batches = math.ceil(len(data) / batch_size)
    for i in range(num_batches):
        if i < num_batches - 1:
            data_slice = data[(batch_size * i):(batch_size * (i + 1))]
            data_indices_slice = data_indices[(batch_size * i):(batch_size * (i + 1))]
        else:
            data_slice = data[(batch_size * i):]
            data_indices_slice = data_indices[(batch_size * i):]
        scaled_data = scale_and_standardize_data(data_slice)
        subsampling_result = faiss_ivf_subsample(scaled_data, cutoff_sig)
        subsampled_data_indices += data_indices_slice[subsampling_result]

    subsampled_data_indices = np.array(subsampled_data_indices)
    subsampled_data = data[subsampled_data_indices]

    if subsampled_data.size == data.size:
        # No more results can be subsampled with the given cutoff and the batch size
        # is < the number of points remaining
        return data

    if len(subsampled_data_indices) > batch_size:
        # recursively subsample, further splitting into batches
        return batch_subsampling(subsampled_data, batch_size, standard_scale, cutoff_sig,
                                 verbose, shuffle, data_indices=subsampled_data_indices)
    else:
        # perform ccsubsample on the combined results from all the batches
        scaled_data = scale_and_standardize_data(subsampled_data)
        subsampling_result = faiss_ivf_subsample(scaled_data, cutoff_sig)
        return subsampled_data_indices[subsampling_result]


def batch_subsampling_with_PCA(
    data,
    batch_size=3000000,
    max_component=30,
    target_variance=0.999999,
    standard_scale=True,
    cutoff_sig=0.2,
    verbose=1,
    shuffle=True,
    data_indices=None
):
    
    """
    Subsample in batches (with PCA pre-processing)
    This is to save the memory if the data set of interest is too large.
    The data set will first be broken down into equally sized batches (defined by batch size)
    that will be subsampled individually.
    The resulting subsampled datapoints will then be pooled together for an overall subsample
    If the final pooled set of subsampled datapoints is larger than batch_size,
    batch_subsampling_with_PCA will be called recursively on that set of datapoints
    Parameters
    -------------
    data: List. the original list of data points
    batch_size [1000000]: Int. the number of datapoints in each batch
    standard_scale [True]: Boolean. Whether to apply standard scaler to the dataset prior to ccsubsample
    cutoff_sig [0.02]: Float. cutoff significance. the cutoff distance equals to the Euclidean
                       norm of the standard deviations in all dimensions of the data points
    rate [0.3]: Float. possibility of deletion
    max_component [30]: Int.the maximum number of PCs to be kept,
                        even the target variance has not been reached
    target_variance [0.999999]: Float. the target sum of variance.
    verbose [1]: integer. level of verbosity
    shuffle [True]: Boolean. whether to shuffle the dataset before breaking down into batchs
    Return
    -------------
    sampling_result : the result list of subsampled data points
    """
    if not data_indices:
        data_indices = np.arange(start=0, stop=len(data))
    subsampled_data_indices = []
    
    if shuffle:
        random.shuffle(data)
    
    num_batches = math.ceil(len(data) / batch_size)
    for i in range(num_batches):
        if i < num_batches - 1:
            data_slice = data[(batch_size * i):(batch_size * (i + 1))]
            data_indices_slice = data_indices[(batch_size * i):(batch_size * (i + 1))]
        else:
            data_slice = data[(batch_size * i):]
            data_indices_slice = data_indices[(batch_size * i):]
        reduced_data = reduce_dimensions_with_pca(data_slice, max_component, target_variance)
        scaled_data = scale_and_standardize_data(reduced_data)
        subsampling_result = faiss_ivf_subsample(scaled_data, cutoff_sig)
        subsampled_data_indices += data_indices_slice[subsampling_result]

    subsampled_data_indices = np.array(subsampled_data_indices)
    subsampled_data = data[subsampled_data_indices]
    
    if subsampled_data.size == data.size:
        # No more results can be subsampled with the given cutoff and the batch size
        # is < the number of points remaining
        return data
    
    if len(subsampled_data_indices) > batch_size:
        # recursively subsample, further splitting into batches
        return batch_subsampling_with_PCA(subsampled_data, batch_size, max_component, target_variance, standard_scale,
                                          cutoff_sig, verbose, shuffle, data_indices=subsampled_data_indices)
    else:
        # perform ccsubsample on the combined results from all the batches
        reduced_data = reduce_dimensions_with_pca(subsampled_data, max_component, target_variance)
        scaled_data = scale_and_standardize_data(reduced_data)
        subsampling_result = faiss_ivf_subsample(scaled_data, cutoff_sig)
        return subsampled_data_indices[subsampling_result]

def imagewise_subsampling(torchg_data, cutoff):
    """
    First subsample each image, then combine whatever points are left and
    subsample again
    :param torchg_data:
    :param cutoff:
    :param extract_img_func:
    :return:
    """
    print("Image-wise sampling")
    imagewise_fingerprints = []
    datapoint_to_image_index = []
    for i, image in enumerate(torchg_data):
        scaled_reduced_data = scale_and_standardize_data(image.fingerprint)
        subsampled_fingerprint = faiss_ivf_subsample(
            scaled_reduced_data,
            cutoff_sig=cutoff,
            verbose=2
        )
        imagewise_fingerprints += list(subsampled_fingerprint)
        print("Image {}: Kept {} datapoints".format(i, len(subsampled_fingerprint)))
        datapoint_to_image_index.extend([i] * len(subsampled_fingerprint))
    
    reduced_dim_fingerprints = reduce_dimensions_with_pca(data=np.array(imagewise_fingerprints))
    data_indices_to_keep = faiss_ivf_subsample(data=reduced_dim_fingerprints,
                                            cutoff_sig=cutoff,
                                            verbose=2)
    image_indices_to_keep = get_image_indices_to_keep(data_indices_to_keep, datapoint_to_image_index)
    return data_indices_to_keep, image_indices_to_keep
