"""Collection of functions to process mini batches."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def invert_full_matrix_np(full_adjacency):
    full_adjacency = np.squeeze(full_adjacency)
    n_nodes = full_adjacency.shape[1]
    full_adjacency = np.append(np.zeros([1, n_nodes]), full_adjacency, axis=0)
    full_adjacency[0, 0] = 1
    adjacency = np.eye(n_nodes) - np.linalg.inv(full_adjacency)
    return adjacency[1:, :]


def batch_symmetrize_np(input_matrix, batch_size, n_nodes):
    """
    Take an n_nodes - 1 x n_nodes matrix and symmetrizes it.
    It concatenates a row of zeros with the matrix,
    adds the transpose and then removes the padded row.
    Parameters
    ----------
    input_matrix: theano tensor
        batch_size x n_nodes - 1 x n_nodes
    batch_size: int
        batch size
    n_nodes: int
        number of nodes of the matrix
    """
    input_matrix = np.concatenate([np.zeros(shape=[batch_size, 1, n_nodes]),
                                  input_matrix], axis=1)
    result = np.zeros(shape=[batch_size, n_nodes, n_nodes])

    for n in range(input_matrix.shape[0]):
        result[n, :, :] = np.squeeze(input_matrix[n, :, :]) + \
                          np.squeeze(input_matrix[n, :, :].T)
    return result[:, 1:, :]


def full_matrix_np(adjacency, n_nodes):
    return np.linalg.inv(np.eye(n_nodes) - adjacency)


def masked_softmax_full_np(input_data):
    batch_size = input_data.shape[0]
    n_nodes = input_data.shape[2]
    output_data = np.append(np.zeros([batch_size, 1, n_nodes]),
                            input_data, axis=1)
    for i in range(batch_size):
        output_data[i, :, :] = \
            full_matrix_np(np.squeeze(output_data[i, :, :]), n_nodes)
    return output_data[:, 1:, :]


def features(X_parent, X_locations):
    """
    Get the features of the dataset.
    Parameters
    ----------
    X_parent: an array of size (batch_size x n_nodes - 1 x n_nodes)
        the adjacency of each matrix.
    X_locations: an array of size (batch_size x n_nodes - 1 x 3)
        the locations of each nodes.
    Returns
    -------
    X_features: an array of size (batch_size x n_nodes x n_features)
        The features currently supports:
            - The adjacency
            - The full adjacency
            - locations
            - distance from immediate parents
    """
    batch_size = X_parent.shape[0]
    n_nodes = X_parent.shape[2]
    X_adjacency = np.append(np.zeros([batch_size, 1, n_nodes]),
                            X_parent, axis=1)
    X_locations = np.append(np.zeros([batch_size, 1, 3]),
                            X_locations,
                            axis=1)

    X_full_adjacency = np.zeros([batch_size, n_nodes, n_nodes])
    X_distance = np.zeros([batch_size, n_nodes, 3])

    for sample in range(batch_size):
        X_full_adjacency[sample, :, :] = \
            full_matrix_np(np.squeeze(X_adjacency[sample, :, :]), n_nodes)
        X_distance[sample, :, :] = \
            np.dot(np.eye(n_nodes) - np.squeeze(X_adjacency[sample, :, :]),
                   np.squeeze(X_locations[sample, :, :]))

    X_features = np.append([X_adjacency,
                            X_full_adjacency,
                            X_locations,
                            X_distance], axis=2)
    return X_features


def get_batch(X_parent_cut, batch_size, n_nodes):
    """
    Make a batch of morphological and geometrical data.
    Parameters
    -----------
    training_data: dict of dicts
        each inner dict is an array
        'geometry': 3-d arrays (locations)
            n_samples x n_nodes - 1 x 3
        'morphology': 2-d arrays
            n_samples x n_nodes - 1 (parent sequences)
        example: training_data['geometry']['n20'][0:10, :, :]
                gives the geometry for the first 10 neurons
                training_data['geometry']['n20'][0:10, :]
                gives the parent sequences for the first 10 neurons
                here, 'n20' indexes a key corresponding to
                20-node downsampled neurons.
    batch_size: int
         batch size.
     batch_counter: the index of the selected batches
         the data for batch are selected from the index
         (batch_counter - 1) * batch_size to
         batch_counter * batch_size of whole data.
     n_nodes: int
         subsampled resolution of the neurons.
    Returns
    -------
    X_locations_real: an array of size (batch_size x n_nodes - 1 x 3)
        the location of the nodes of the neuorns.
    X_parent_real: an array of size (batch_size x n_nodes x n_nodes - 1)
        the parent sequence for parent of the neuron.
    """
    enc = OneHotEncoder(n_values=n_nodes)

    X_parent_real = np.reshape(enc.fit_transform(X_parent_cut).toarray(),
                               [batch_size, n_nodes - 1, n_nodes])
    return X_parent_real


def gen_batch(geom_model,
              morph_model,
              conditioning_rule='mgd',
              batch_size=64,
              n_nodes=20,
              input_dim=100):
    """
    Generate a batch of samples from generators.
    Parameters
    ----------
    geom_model: list of keras objects
        geometry generator
    morph_model: list of keras objects
        morphology generator
    conditioning_rule: str
        'mgd': P_w(disc_loss|g,m) P(g|m) P(m)
        'gmd': P_w(disc_loss|g,m) P(m|g) P(g)
    batch_size: int
        batch size
    n_nodes: list of ints
        number of nodes
    input_dim: int
        dimensionality of noise input
    Returns
    -------
    locations: float (batch_size x 3 x n_nodes - 1)
        batch of generated locations
    parent: float (batch_size x n_nodes x n_nodes - 1)
        batch of generated morphology
    """
    locations = None
    parent = None


    # Generate noise code
    noise_code = np.random.rand(batch_size, 1, input_dim)

    # Generate geometry and morphology
    if conditioning_rule == 'mgd':
        parent = morph_model.predict(noise_code)
        locations = cond_geom_model.predict([noise_code,
                                                parent])
    elif conditioning_rule == 'gmd':
        locations = geom_model.predict(noise_code)
        parent = cond_morph_model.predict([noise_code,
                                              locations])
    elif conditioning_rule == 'none':
        locations = geom_model.predict(noise_code)
        parent = morph_model.predict(noise_code)

    return locations, parent
