"""Collection of useful data transforms."""

# Imports
import numpy as np
import Neuron

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from numpy.linalg import inv


def get_leaves(nodes, parents):
    """
    Compute the list of leaf nodes.
    Parameters
    ----------
    nodes: list
        list of all nodes in the tree
    parents: list
        list of parents for each node
    Returns
    -------
    leaves: list
        sorted list of leaf nodes
    """
    leaves = np.sort(list(set(nodes) - set(parents)))
    return leaves


def encode_prufer(parents, verbose=0):
    """
    Convert the parents sequence to a prufer sequence.
    Parameters
    ----------
    parents: list
        list of parents for each node
    verbose: bool
        default is False
    Returns
    -------
    prufer: list
        corresponding prufer sequence
    """
    n_nodes = len(parents)
    nodes = range(n_nodes)

    prufer = list()
    for n in range(n_nodes - 2):

        # Recalculate all the leaves
        leaves = get_leaves(nodes, parents)
        if verbose:
            print 'leaves', leaves

        # Add the parent of the lowest numbered leaf to the sequence
        leaf_idx = np.where(nodes == leaves[0])[0][0]
        prufer.append(parents[leaf_idx])
        if verbose:
            print 'prufer', prufer

        # Remove the lowest numbered leaf and its corresponding parent
        del nodes[leaf_idx]
        del parents[leaf_idx]

        if verbose:
            print 'nodes', nodes
            print 'parents', parents
            print 60*'-'

    return prufer


def decode_prufer(prufer, verbose=0):
    """
    Convert the prufer sequence to a parents sequence.
    Parameters
    ----------
    prufer: list
        prufer sequence
    verbose: bool
        default is False
    Returns
    -------
    parents: list
        corresponding list of parents for each node
    """
    n_nodes = len(prufer) + 2
    n_prufer = len(prufer)
    nodes = range(n_nodes)
    parents = -1 * np.ones(n_nodes)

    for n in range(n_prufer):
        if verbose:
            print nodes
            print prufer
        leaves = list(get_leaves(nodes, prufer))
        k = leaves[0]
        j = prufer[0]

        if k == 0:
            k = leaves[1]

        if verbose:
            print k, j
        parents[k] = j

        leaf_idx = np.where(nodes == k)[0][0]
        del nodes[leaf_idx]
        del prufer[0]

        if verbose:
            print 60*'-'

    parents[nodes[1]] = nodes[0]
    return list(parents.astype(int))


def reordering_prufer(parents, locations):
    """
    Reorder a given parents sequence.
    Parent labels < children labels.
    Parameters
    ----------
    parents: numpy array
        sequence of parents indices
        starts with -1
    locations: numpy array
        n - 1 x 3
    Returns
    -------
    parents_reordered: numpy array
        sequence of parents indices
    locations_reordered: numpy array
        n - 1 x 3
    """
    length = len(parents)

    # Construct the adjacency matrix
    adjacency = np.zeros([length, length])
    adjacency[parents[1:], range(1, length)] = 1

    # Discover the permutation with Schur decomposition
    full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
    full_adjacency_permuted, permutation_matrix = \
        scipy.linalg.schur(full_adjacency)

    # Reorder the parents
    parents_reordered = \
        np.argmax(np.eye(length) - np.linalg.inv(full_adjacency_permuted),
                  axis=0)
    parents_reordered[0] = -1

    # Reorder the locations
    locations = np.append([[0., 0., 0.]], locations, axis=0)
    locations_reordered = np.dot(permutation_matrix, locations)

    return parents_reordered, locations_reordered[1:, :]


def swc_to_neuron(matrix):
    """
    Return the Neuron object from swc matrix.
    Parameters
    ----------
    matrix: numpy array
        numpy array of the size n_nodes*7.
    Return
    ------
    Neuron: Neuron
        a neuron obj with the given swc format.
    """
    return Neuron(file_format='Matrix of swc', input_file=matrix)


def downsample_neuron(neuron,
                      method='random',
                      number=30):
    """
    Downsampling neuron with different methods.
    Parameters
    ----------
    neuron: Neuron
        given neuron to subsample.
    number: int
        the number of subsamling.
    method: str
        the methods to subsample. It can be: 'random', 'regularize','prune',
        'strighten', 'strighten-prune'.
    Return
    ------
    Neuron: Neuron
        a subsampled neuron with given number of nodes.
    """
    if(method == 'random'):
        return subsample.random_subsample(neuron, number)


def get_data(neuron_database, method, subsampling_numbers):
    """
    Preparing data for the learning.
    Parameters
    ----------
    neuron_database: list
        the elements of the list are Neuron obj.
    method: str
        the method to subsample.
    subsampling_numbers: array of int
        The range of number to subsample.
    Returns
    -------
    data: dic
        a dic of two classes: 'morphology' and 'geometry'.
        'geometry' is a list of size sampling_division. The i-th element of the
        list is an array of size (datasize* n_nodes - 1*3).
        'morphology' is a list of size sampling_division. The i-th element of
        the list is an array of size (datasize* n_nodes* n_nodes -2).
    """
    l = len(neuron_database)
    morph = np.zeros([l, subsampling_numbers - 2])
    geo = np.zeros([l, subsampling_numbers - 1, 3])
    data = dict()
    for i in range(l):
        sub_neuron = downsample_neuron(neuron=neuron_database[i],
                                       method=method,
                                       number=subsampling_numbers)
        par = sub_neuron.parent_index
        par[0] = -1
        morph[i, :] = encode_prufer(par.tolist())
        geo[i, :, :] = sub_neuron.location[:, 1:].T

    data['morphology'] = dict()
    data['morphology']['n'+str(subsampling_numbers)] = morph
    data['geometry'] = dict()
    data['geometry']['n'+str(subsampling_numbers)] = geo
    return data


def make_swc_from_prufer_and_locations(data):
    # the prufer code and the location are given.
    parents_code = np.array(decode_prufer(list(data['morphology'])))
    location = data['geometry']
    M = np.zeros([len(parents_code), 7])
    M[:, 0] = np.arange(1, len(parents_code)+1)
    M[0, 1] = 1
    M[1:, 1] = 2
    M[1:, 2:5] = location
    parents_code[1:] = parents_code[1:] + 1
    M[:, 6] = parents_code
    return Neuron(file_format='Matrix of swc', input_file=M)
