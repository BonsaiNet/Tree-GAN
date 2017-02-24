"""Collection of custom Keras layers."""

# Imports
from keras import backend as K
from keras.layers.core import Dense, Reshape, RepeatVector, Lambda, Dropout
from keras.layers import Input, merge
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


# Apply batch symmetrization (A + A.T)
def batch_symmetrize(input_matrix, batch_size, n_nodes):
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
    input_matrix = K.concatenate([K.zeros(shape=[batch_size, 1, n_nodes]),
                                  input_matrix], axis=1)
    result, updates = \
        K.theano.scan(fn=lambda n: input_matrix[n, :, :] +
                      input_matrix[n, :, :].T,
                      sequences=K.arange(input_matrix.shape[0]))
    return result[:, 1:, :]


# Masked softmax Lambda layer
def masked_softmax(input_layer, n_nodes, batch_size):
    """
    A Lambda layer to mask a matrix of outputs to be lower-triangular.
    Each row must sum up to one. We apply a lower triangular mask of ones
    and then add an upper triangular mask of a large negative number.
    Parameters
    ----------
    input_layer: keras layer object
        (n x 1, n) matrix
    n_nodes: int
        number of nodes
    batch_size: int
        batch size
    Returns
    -------
    output_layer: keras layer object
        (n x 1, n) matrix
    """
    # input_layer = batch_symmetrize(input_layer, batch_size, n_nodes)
    mask_lower = K.theano.tensor.tril(K.ones((n_nodes - 1, n_nodes)))
    mask_upper = \
        K.theano.tensor.triu(-100. * K.ones((n_nodes - 1, n_nodes)), 1)
    mask_layer = mask_lower * input_layer + mask_upper
    mask_layer = mask_layer + 0 * K.eye(n_nodes)[0:n_nodes - 1, 0:n_nodes]
    mask_layer = \
        K.reshape(mask_layer, (batch_size * (n_nodes - 1), n_nodes))
    softmax_layer = K.softmax(mask_layer)
    output_layer = K.reshape(softmax_layer, (batch_size, n_nodes - 1, n_nodes))
    return output_layer


# Compute full adjacency matrix
def full_matrix(adjacency, n_nodes):
    """
    Returning the full adjacency matrix of adjacency.
    Parameters
    ----------
    adjacency: keras layer object
        (n , n) matrix
    Returns
    -------
    keras layer object
        (n , n) matrix
    """
    return K.theano.tensor.nlinalg.matrix_inverse(K.eye(n_nodes) - adjacency)


def batch_full_matrix(adjacency, n_nodes, batch_size):
    result, updates = \
        K.theano.scan(fn=lambda n: full_matrix(adjacency[n, :, :], n_nodes),
                      sequences=K.arange(batch_size))
    return result


# Masked softmax Lambda layer
def masked_softmax_full(input_layer, n_nodes, batch_size):
    """
    A Lambda layer to compute a lower-triangular version of the full adjacency.
    Each row must sum up to one. We apply a lower triangular mask of ones
    and then add an upper triangular mask of a large negative number.
    After that we return the full adjacency matrix.
    Parameters
    ----------
    input_layer: keras layer object
        (n x 1, n) matrix
    Returns
    -------
    output_layer: keras layer object
        (n x 1, n) matrix
    """
    mask_layer = masked_softmax(input_layer, n_nodes, batch_size)
    mask_layer = \
        K.concatenate([K.zeros(shape=[batch_size, 1, n_nodes]), mask_layer],
                      axis=1)
    result, updates = \
        K.theano.scan(fn=lambda n: full_matrix(mask_layer[n, :, :], n_nodes),
                      sequences=K.arange(batch_size))
    return result[:, 1:, :]


def distance_from_parent(adjacency, locations, n_nodes, batch_size):
    """
    Return distance from parent.
    Parameters
    ----------
    adjacency: theano/keras tensor
        (batch_size x n_nodes - 1 x n_nodes) matrix
    locations: theano/keras tensor
        (batch_size x n_nodes x 3) matrix
    Returns
    -------
    result: keras layer object
        (batch_size x n_nodes - 1 x n_nodes) matrix
    """
    result, updates = \
        K.theano.scan(fn=lambda n: K.dot(K.eye(n_nodes) - adjacency[n, :, :],
                                         locations[n, :, :]),
                      sequences=K.arange(batch_size))
    # result, updates = \
    #     K.theano.scan(fn=lambda n: K.dot(adjacency[n, :, :],
    #                                      locations[n, :, :]),
    #                   sequences=K.arange(batch_size))
    return result

def locations_by_distance_from_parent(full_adjacency, distance_from_parent, batch_size):
    """
    Return distance from parent.
    Parameters
    ----------
    full_adjacency: theano/keras tensor
        (batch_size x n_nodes x n_nodes) matrix
    distance_from_parent: theano/keras tensor
        (batch_size x n_nodes x 3) matrix
    Returns
    -------
    result: keras layer object
        (batch_size x n_nodes - 1 x n_nodes) matrix
    """
    result, updates = \
        K.theano.scan(fn=lambda n: K.dot(full_adjacency[n, :, :],
                                         distance_from_parent[n, :, :]),
                      sequences=K.arange(batch_size))
    return result

def feature_extractor(inputs,
                      n_nodes,
                      batch_size):
    """
    Compute various features and concatenate them.
    Parameters
    ----------
    morphology_input: keras layer object
        (batch_size x n_nodes - 1 x n_nodes)
        the adjacency matrix of each sample.
    geometry_input: keras layer object
        (batch_size x n_nodes - 1 x 3)
        the locations of each nodes.
    n_nodes: int
        number of nodes
    batch_size: int
        batch size
    Returns
    -------
    features: keras layer object
        (batch_size x n_nodes x n_features)
        The features currently supports:
            - The adjacency
            - The full adjacency
            - locations
            - distance from imediate parents
    """
    geometry_input = inputs[:, :, :3]
    morphology_input = inputs[:, :, 3:]

    adjacency = \
        K.concatenate([K.zeros(shape=(batch_size, 1, n_nodes)),
                       morphology_input], axis=1)

    full_adjacency = \
        batch_full_matrix(adjacency, n_nodes, batch_size)
    geometry_input = K.concatenate([K.zeros(shape=(batch_size, 1, 3)),
                                    geometry_input], axis=1)

    # distance = distance_from_parent(adjacency,
    #                                 geometry_input,
    #                                 n_nodes,
    #                                 batch_size)

    # distance = locations_by_distance_from_parent(full_adjacency=full_adjacency,
    #                                              distance_from_parent=geometry_input,
    #                                              batch_size=batch_size)
    #
    filled_full_adjacency_x = \
        full_adjacency*K.repeat_elements(K.expand_dims(geometry_input[:,:,0],2),n_nodes, axis=2)
    filled_full_adjacency_y = \
        full_adjacency*K.repeat_elements(K.expand_dims(geometry_input[:,:,1],2),n_nodes, axis=2)
    filled_full_adjacency_z = \
        full_adjacency*K.repeat_elements(K.expand_dims(geometry_input[:,:,2],2),n_nodes, axis=2)

    features = K.concatenate([adjacency,
                              full_adjacency,
                              geometry_input,
                              filled_full_adjacency_x,
                              filled_full_adjacency_y,
                              filled_full_adjacency_z], axis=2)
    return features
