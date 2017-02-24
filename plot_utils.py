"""Collection of functions to visualization generated samples."""

# Imports
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Local imports
import data_transforms
import Neuron
import visualization


def plot_example_neuron_from_parent(X_locations, X_parent):
    """
    Show an example neuron.

    Parameters
    ----------
    X_locations: array of shape (n-1 ,3)
        the locations of `n` nodes, except the root.
    X_parent: array of shape (n-1 ,n)
        the adjacency matrix or lower-triangular matrix

    Returns
    -------
    neuron_object: the Neuron
        with the locations of `X_locations` and parent of argmax of `X_parent`
    plot: the 2D plotting of neuron
    """

    locations = np.squeeze(X_locations)
    parent = np.squeeze(X_parent).argmax(axis=1) + 1
    full = np.zeros([parent.shape[0]+1, parent.shape[0] + 1])
    full[range(1, parent.shape[0]+1), parent-1] = 1
    full = LA.inv(np.eye(parent.shape[0]+1) - full)
    locations = np.dot(full, np.append(np.zeros([1, 3]), locations, axis=0))

    # buliding swc matrix
    M = np.zeros([parent.shape[0] + 1, 7])
    M[:, 0] = np.arange(1, parent.shape[0] + 2)
    M[0, 1] = 1
    M[1:, 1] = 2
    M[:, 2:5] = locations
    M[1:, 6] = parent
    M[0, 6] = -1
    neuron_object = Neuron.Neuron(file_format='Matrix of swc', input_file=M)

    visualization.plot_2D(neuron_object)
    plt.show()

    return neuron_object


def plot_adjacency(X_parent_real, X_parent_gen):
    """
    Plot a pair of adjacency matrices side by side.

    stuff.
    """
    for sample in range(X_parent_real.shape[0]):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(X_parent_real[sample, :, :],
                   interpolation='none',
                   cmap='Greys')
        plt.subplot(1, 2, 2)
        plt.imshow(X_parent_gen[sample, :, :],
                   interpolation='none',
                   cmap='Greys')


def plot_loss_trace(loss):
    """
    Plot trace of loss.

    Parameters
    ----------
    loss: list or array of a loss trace
    """
    plt.figure(figsize=(3, 2))
    plt.plot(loss)
    plt.show()


def plot_example_neuron_from_prufer(X_locations, X_prufer):
    """
    Show an example neuron.

    stuff.
    """
    locations = np.squeeze(X_locations)
    prufer = np.squeeze(X_prufer).argmax(axis=1)

    soma = np.array([[0., 0., 0.]])
    np.append(soma, np.squeeze(locations), axis=0)

    parents = np.array(data_transforms.decode_prufer(list(prufer)))
    parents_reordered, locations_reordered = \
        data_transforms.reordering_prufer(parents, np.squeeze(locations))

    prufer_reordered = data_transforms.encode_prufer(list(parents_reordered))

    input_code = dict()
    input_code['morphology'] = np.array(prufer_reordered)
    input_code['geometry'] = np.squeeze(locations_reordered)
    neuron_object = \
        data_transforms.make_swc_from_prufer_and_locations(input_code)

    Neuron.visualization.plot_2D(neuron_object)
    return neuron_object
