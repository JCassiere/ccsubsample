# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:36:48 2017
@author: Xiangyun Lei
"""

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

try:
    import cPickle as pickle
except:
    import pickle


def sqrt_of_summed_variance(data):
    """
    Find the variance for each dimension across all inputs, then sum them and take
    the square root
    :param data: List
    :return:
    """
    array_data = np.asarray(data)
    std = np.std(data, axis=0)
    variances = np.power(std, 2)
    sqrt_summed_variance = np.sqrt(np.sum(variances))
    return sqrt_summed_variance


def pykdtree_query(full_data, data_slice):
    kd_tree = KDTree(full_data, leafsize=6)
    return kd_tree.query(data_slice, k=2)

def single_process_kdtree_query(remaining_datapoints):
    kd_tree = KDTree(remaining_datapoints, leafsize=6)
    distances, indices = kd_tree.query(remaining_datapoints, k=2)
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


def kdtree_subsample(data, cutoff_sig=0.25, verbose=1):
    """
    Using Nearest-Neighbor search based algorithm, find the list of indices of the subsampled dataset
    Parameters
    -------------
    :param data: the list of data to subsample
    :param cutoff_sig: float -  cutoff significance. the cutoff distance equals to the Euclidean
        norm of the standard deviations in all dimensions of the data points
    :param verbose: int - level of verbosity
    :param num_cpus_to_not_use: int - the number of machine cpus to leave free in the case of
        using multiprocessing to query the kdtree
    Return
    -------------
    overall_keep_list: The list of indices of the final subsampled entries
    """
    start = 0
    
    if verbose >= 1:
        start = time.time()
        print("Started NN-ccsubsample, original length: {}".format(len(data)))
    
    cutoff = cutoff_sig * sqrt_of_summed_variance(data)
    
    # initialize the index
    original_data_indices = np.arange(len(data))
    permanent_keep_indices = set()
    
    remaining_datapoints = np.asarray(data)
    keep_going = True
    iter_count = 1
    old_overall_keep_len = 0
    iter_start = 0
    while keep_going:
        if verbose >= 2:
            print("Start iteration {}, Sampleable data points remaining: {}".format(iter_count, len(original_data_indices)))
            iter_start = time.time()
        
        # build and query nearest neighbour model
        # if remaining_datapoints.size < 5000000:
        distances, indices = single_process_kdtree_query(remaining_datapoints)
        # else:
        #     distances, indices = multi_process_kdtree_query(remaining_datapoints, num_cpus_to_not_use)
        
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
        
        # keep_indices length can be 0 if all remaining points have been added to the
        # permanent keep list
        if len(keep_indices) == 0:
            break
        original_data_indices = original_data_indices[keep_indices]
        remaining_datapoints = remaining_datapoints[keep_indices]
        
        overall_keep_len = original_data_indices.size + len(permanent_keep_indices)
        if overall_keep_len == old_overall_keep_len:
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


def subsample_clustering(data, start_cutoff_sig=0.2, max_clusters=20, num_cpus_to_not_use=1):
    """
    Extend the nearest-neighbor ccsubsample algorithm (as seen in subsample() above) to a fast
    clustering algorithm. When a point is "removed", instead assign it (and all its "cluster" points)
    to its nearest neighbor's "cluster". If a point has a cluster attached to it, it is termed a "leader".
    This is similar to agglomerative clustering where the distance metric is distance between cluster
    "leaders" rather than average linkage, etc.
    -------------
    :param data: the list of data to subsample
    :param start_cutoff_sig: float -  cutoff significance. the cutoff distance equals to the Euclidean
        norm of the standard deviations in all dimensions of the data points
    :param max_clusters: the maximum desired number of clusters
    :param num_cpus_to_not_use: int - the number of machine cpus to leave free in the case of
        using multiprocessing to query the kdtree
    :return clusters: dict[(int, list)] A dictionary where the keys are the original datapoint indices of cluster
        "leaders", and the keys are lists of datapoint indices belonging to that "leader's" cluster
    """
    start = 0
    
    cutoff = start_cutoff_sig * sqrt_of_summed_variance(data)
    
    # initialize the index
    original_data_indices = np.arange(len(data))
    permanent_keep_indices = set()
    
    remaining_datapoints = np.array(data)
    keep_going = True
    iter_count = 1
    old_overall_keep_len = 0
    iter_start = 0

    clusters = {}
    for i in range(remaining_datapoints.shape[0]):
        clusters[i] = [i]
        
    # heuristic for testing whether enough clustering has been done so far
    max_singleton_clusters = remaining_datapoints.shape[0] * 0.25
    
    # we want meaningful clusters, so don't count singleton clusters (i.e. outlier data points)
    # or small clusters (with <5 members) when checking if we are below max_clusters
    while num_large_clusters(clusters) > max_clusters or num_small_clusters(clusters) > max_singleton_clusters:
        while keep_going:
            # build and query nearest neighbour model
            # if remaining_datapoints.size < 5000000:
            distances, indices = single_process_kdtree_query(remaining_datapoints)
            # else:
            #     distances, indices = multi_process_kdtree_query(remaining_datapoints, num_cpus_to_not_use)
            
            # if distance between a point and its nearest neighbor is below cutoff distance,
            # add the pair's indices (for this iteration) to the candidate removal list
            removal_candidate_indices_with_neighbor = indices[:][distances[:, 1] <= cutoff]
            
            # if distance between a point and its nearest neighbor is above the cutoff distance,
            # the former point can never be removed, so add it to the permanent keep list
            iteration_permanent_keeps = original_data_indices[distances[:, 1] > cutoff]
            
            # set aside any data points above the cutoff, since they can never be removed
            permanent_keep_indices = permanent_keep_indices.union(list(iteration_permanent_keeps))
            
            keep_remove_pairs = find_keep_remove_pairs(removal_candidate_indices_with_neighbor)
            # keep_indices length can be 0 if all remaining points have been added to the
            # permanent keep list
            keep_indices = find_keep_indices(removal_candidate_indices_with_neighbor, keep_remove_pairs)
            if len(keep_indices) == 0:
                break

            keep_remove_array = np.array(keep_remove_pairs)
            keep_original = np.expand_dims(original_data_indices[keep_remove_array[:, 0]], axis=1)
            remove_original = np.expand_dims(original_data_indices[keep_remove_array[:, 1]], axis=1)
            original_keep_remove_pair_indices = np.concatenate((keep_original, remove_original), axis=1)
            clusters = update_clusters(original_keep_remove_pair_indices, clusters)
            
            original_data_indices = original_data_indices[keep_indices]
            remaining_datapoints = remaining_datapoints[keep_indices]
            
            overall_keep_len = original_data_indices.size + len(permanent_keep_indices)
            if overall_keep_len == old_overall_keep_len:
                keep_going = False
            
            old_overall_keep_len = overall_keep_len
        # make the cutoff bigger in order to merge more clusters
        original_data_indices = np.array(sorted(clusters.keys()))
        remaining_datapoints = np.array(data)[original_data_indices]
        
        cutoff *= 1.25
    return clusters


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
        subsampling_result = kdtree_subsample(scaled_data, cutoff_sig)
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
        subsampling_result = kdtree_subsample(scaled_data, cutoff_sig)
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
        subsampling_result = kdtree_subsample(scaled_data, cutoff_sig)
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
        subsampling_result = kdtree_subsample(scaled_data, cutoff_sig)
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
        subsampled_fingerprint = kdtree_subsample(
            scaled_reduced_data,
            cutoff_sig=cutoff,
            verbose=2
        )
        imagewise_fingerprints += list(subsampled_fingerprint)
        print("Image {}: Kept {} datapoints".format(i, len(subsampled_fingerprint)))
        datapoint_to_image_index.extend([i] * len(subsampled_fingerprint))
    
    reduced_dim_fingerprints = reduce_dimensions_with_pca(data=np.array(imagewise_fingerprints))
    data_indices_to_keep = kdtree_subsample(data=reduced_dim_fingerprints,
                                            cutoff_sig=cutoff,
                                            verbose=2)
    image_indices_to_keep = get_image_indices_to_keep(data_indices_to_keep, datapoint_to_image_index)
    return data_indices_to_keep, image_indices_to_keep
