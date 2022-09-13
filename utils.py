import json
import os
import pickle
import numpy as np
import time
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_fingerprints_with_image_indices(pytorch_geom_data):
    """
    Extracts individual fingerprints from images
    :param pytorch_geom_data: List[Data] - a list of torch_geometric Data,
        where each element is an image from the input dataset
    :return: flattened_fingerprints: np.ndarray[float]
             fingerprint_to_image_index: np.ndarray[int]
       index i of flattened_fingerprints corresponds to index i of fingerprint_to_image_index
    """
    per_image_fingerprints = []
    fingerprint_to_image_index = []
    
    for i, torch_image in enumerate(pytorch_geom_data):
        image_fingerprint = torch_image.fingerprint.numpy()
        per_image_fingerprints.append(image_fingerprint)
        fingerprint_to_image_index += [i for _ in range(image_fingerprint.shape[0])]
    
    flattened_fingerprints = np.vstack(per_image_fingerprints)
    fingerprint_to_image_index = np.array(fingerprint_to_image_index)
    
    return flattened_fingerprints, fingerprint_to_image_index


def extract_energies(pytorch_geom_data):
    """
    Extracts the energy for each image
    :param pytorch_geom_data: List[Data] - a list of torch_geometric Data,
        where each element is an image from the input dataset
    :return: energy_list List[float] - a list with an energy for each image
    """
    energy_list = [image.energy.tolist() for image in pytorch_geom_data]
    
    return energy_list


def average_images(pytorch_geom_data):
    """
    Get a per-image average value
    :param pytorch_geom_data: List[Data] - a list of torch_geometric Data,
        where each element is an image from the input dataset
    :return: image_averages: np.ndarray[float] - an ndarray of the average value for each input image
             image_indices: np.ndarray[int] - an ndarray containing the indices for the images
    """
    image_averages = np.array([image.fingerprint.numpy().mean() for image in pytorch_geom_data])
    image_indices = np.array(list(range(len(image_averages))))
    
    return image_averages, image_indices


def average_images_over_fingerprints(pytorch_geom_data):
    """
    Get a per-image average fingerprint
    :param pytorch_geom_data: List[Data] - a list of torch_geometric Data,
        where each element is an image from the input dataset
    :return: image_averages: np.ndarray[float] - an ndarray of the average fingerprint for each input image
             image_indices: np.ndarray[int] - an ndarray containing the indices for the images
    """
    average_fingerprints = np.array([image.fingerprint.numpy().mean(axis=0) for image in pytorch_geom_data])
    image_indices = np.array(list(range(len(average_fingerprints))))

    return average_fingerprints, image_indices


def get_image_indices_to_keep(data_indices_to_keep, image_indices):
    return np.array(list(set(image_indices[data_indices_to_keep])))


def reduce_dimensions_with_pca(data, max_components=10, target_variance=0.99, verbose=1):
    """
    Reduce dimensions of the fingerprint data using PCA. For input to subsampling algorithm
    :param data: np.ndarray[float] - the fingerprint data which needs its dimensionality reduced
    :param max_components: int - the maximum number of principal components (PCs) to keep
    :param target_variance: float - the desired amount of variance explained by the kept PCs
    :param verbose: int - level of verbosity when printing to stdout
    :return: reduced_fingerprints: np.ndarray[float] - the dimensionally-reduced fingerprints
    """
    start = 0
    if verbose >= 1:
        print("Beginning PCA for dimensionality reduction")
        start = time.time()
    
    pca = PCA(svd_solver="randomized")
    data_pca = pca.fit_transform(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # determine how many PCs to be kept
    num_principal_components = 1
    pc_variance_prefix_sum = np.cumsum(explained_variance_ratio)
    while True:
        sum_explained_variance = pc_variance_prefix_sum[num_principal_components - 1]
        
        if verbose >= 2:
            print_string = "Explained variance for {} principal components: {}"
            print(print_string.format(num_principal_components, sum_explained_variance))
        
        if sum_explained_variance > target_variance or \
                num_principal_components == max_components:
            break
        
        # Abort PCA if it won't end up reducing dimensionality
        if num_principal_components == len(data[0]):
            return data
        
        num_principal_components += 1
    
    if verbose >= 1:
        time_elapsed = time.time() - start
        print_string = "Stopped PCA at {} components. Total explained variance: {}. " \
                       "Time elapsed: {}"
        print(print_string.format(num_principal_components, sum_explained_variance, time_elapsed))
    
    reduced_fingerprints = data_pca[:, :num_principal_components]
    return reduced_fingerprints


def scale_and_standardize_data(data):
    """
    Scale and standardize all dimensions of a set of data to mean 0 and variance 1
    :param data: np.ndarray[float] - The collection of data to scale and standardize
    return: scaled_data: np.ndarray[float] - the scaled and standardized data
    """
    scaled_data = StandardScaler().fit_transform(np.asarray(data))
    return scaled_data


def get_images_hash(images):
    """
    Create a hash for the input images for identification of a subsampling run when
    loading or storing results
    :param images: List[Atom] - The images for which a hash is created
    :return: images_hash: str - the hash of the images
    """
    images_hash = hashlib.md5(str(images).encode("utf-8")).hexdigest()
    return images_hash


def sampler_config_hash(config):
    """
    Create a hash of the sampling config. This hash can be used to save results or
    retrieve previous results
    :param config: dict - dictionary of values describing subsampling configuration
    :return: sampler_hash: str - hashed string representing a description of the subsampling
      parameters used
    """
    config_json = json.dumps(config, sort_keys=True)
    sampler_hash = hashlib.md5(config_json.encode("utf-8")).hexdigest()
    return sampler_hash
    
    
def save_sampling_results(config, base_save_dir, images, subsampled_image_indices):
    """
    Save the results of subsampling along with the configuration used
    to get them
    :param config: dict - dictionary of values describing subsampling configuration
    :param base_save_dir: str - directory that will hold the subsampling results
    :param images: List[Atoms] - the list of images to be used for identifying the
               inputs for the subsampling results
    :param subsampled_image_indices: List[int] - the indices of the data remaining after
               subsampling
    :return: None
    """
    images_hash = get_images_hash(images)
    sampler_hash = sampler_config_hash(config)
    out_dir = "/".join([base_save_dir, sampler_hash, images_hash])
 
    # if dir not exist, make dir
    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)
    
    results_filename = out_dir + "/" + sampler_hash + "_image_indices.pkl"
    
    config_filename = out_dir + "/" + sampler_hash + "_config.json"
    
    with open(results_filename, "wb") as out_file:
        pickle.dump(subsampled_image_indices, out_file)
    
    json.dump(config, open(config_filename, "w+"), indent=2)


def load_if_exists(config, base_save_dir, images):
    """
    Load previous subsampling results if they exist
    :param config: dict - dictionary of values describing subsampling configuration
    :param base_save_dir: str - directory that may hold the subsampling results
    :param images: List[Atoms] - the list of images to be used for identifying the
               inputs for the subsampling results
    :return: kept image indices from a previous subsampling run
    """
    images_hash = get_images_hash(images)
    sampler_hash = sampler_config_hash(config)
    out_dir = "/".join([base_save_dir, sampler_hash, images_hash])
    results_filename = out_dir + "/" + sampler_hash + "_image_indices.pkl"
    results_path = out_dir + "/" + results_filename
    if os.path.isfile(results_path):
        print("Loading sampled indices...")
        with open(results_path, "rb") as pickled_results:
            return pickle.load(pickled_results)
    else:
        return None
