"""Collections of Models for syntetic data."""
import numpy as np


def y_shape(data_size=1000,
            n_nodes=20,
            distance_from_parent=True,
            first_length=None,
            branching_node=None):
    """
    Generating y-shape neurons.

    Parameters
    ----------
    data_size: int
        the size of data, i.e. the number of neurons
    n-nodes: int
        number of nodes for generated neurons
    distance_from_parent: boolean
        if `True` it puts the distance from parent for the locations
        if `False` the 3d coordinate of nodes are placed
    first_length: int
        number of nodes in the positive slop of "y-shape".
        if None it will not constraints on this number and it comes from
        uniform distribution.
    branching_node: int
        the index of branching node
        if None it will not constraints on this position and it selects
        randomly from available postions

    Returns
    -------
    data: dict
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
                 20-node downsampled neurons
    """
    morph = np.zeros([data_size, n_nodes - 1])
    geo = np.zeros([data_size, n_nodes - 1, 3])
    data = dict()
    for i in range(data_size):
        if first_length is None:
            f_lenght = np.floor((n_nodes-2)*np.random.rand())
        else:
            f_lenght = first_length

        if branching_node is None:
            f_lenght = np.array(f_lenght, dtype=float)
            b_node = np.floor(f_lenght*np.random.rand())
        else:
            b_node = branching_node

        f_lenght = np.array(f_lenght, dtype=int)
        b_node = np.array(b_node, dtype=int)

        par = np.append(np.arange(0, f_lenght+1), b_node)
        par = np.append(par, np.arange(f_lenght+2, n_nodes-1))
        par = np.append(-1, par)

        if distance_from_parent is True:
            morph[i, :] = par[1:]
            geo[i, 0, 0:2] = np.random.rand(2)
            for j in range(1, f_lenght+1):
                geo[i, j, 0:2] = np.random.rand(2)
            a = np.random.rand(2)
            a[0] = -a[0]
            geo[i, f_lenght+1, 0:2] = a

            for j in range(f_lenght+2, n_nodes-1):
                a = np.random.rand(2)
                a[0] = -a[0]
                geo[i, j, 0:2] = a

        else:
            morph[i, :] = par[1:]
            geo[i, 0, 0:2] = np.random.rand(2)
            for j in range(1, f_lenght+1):
                geo[i, j, 0:2] = geo[i, j-1, 0:2] + np.random.rand(2)
            a = np.random.rand(2)
            a[0] = -a[0]
            geo[i, first_length+1, 0:2] = geo[i, b_node-1, 0:2] + a

            for j in range(f_lenght+2, n_nodes-1):
                a = np.random.rand(2)
                a[0] = -a[0]
                geo[i, j, 0:2] = geo[i, j-1, 0:2] + a

    data['morphology'] = dict()
    data['morphology']['n'+str(n_nodes)] = morph
    data['geometry'] = dict()
    data['geometry']['n'+str(n_nodes)] = geo

    return data
