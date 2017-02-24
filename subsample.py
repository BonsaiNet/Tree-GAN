"""Collection of subsampling method on the neurons."""
import numpy as np
import Neuron as Neuron
from Neuron import Node as Node
from numpy import linalg as LA


def find_sharpest_fork(nodes):
    """
    Looks at the all branching point in the Nodes list, selects those which
    both its children are end points and finds the closest pair of childern
    (the distance between children).

    Parameters
    ----------
    Nodes: list
    the list of Node

    Returns
    -------
    sharpest_pair: array
        the index of the pair of closest pair of childern
    distance: float
        Distance of the pair of children
    """
    pair_list = []
    Dis = np.array([])
    for n in nodes:
        if n.parent is not None:
            if n.parent.parent is not None:
                a = n.parent.children
                if(isinstance(a, list)):
                    if(len(a)==2):
                        n1 = a[0]
                        n2 = a[1]
                        if(len(n1.children) == 0 and len(n2.children) == 0):
                            pair_list.append([n1 , n2])
                            dis = LA.norm(a[0].xyz - a[1].xyz,2)
                            Dis = np.append(Dis,dis)
    if(len(Dis)!= 0):
        (b,) = np.where(Dis == Dis.min())
        sharpest_pair = pair_list[b[0]]
        distance = Dis.min()
    else:
        sharpest_pair = [0,0]
        distance = 0.
    return sharpest_pair, distance




def random_subsample(neuron, num):
    """
    randomly selects a few nodes from neuron and builds a new neuron with them. The location of these node in the new neuron
    is the same as the original neuron and the morphology of them is such that if node A is parent (or grand parent) of node B
    in the original neuron, it is the same for the new neuron.

    Parameters
    ----------
    num: int
        number of nodes to be selected randomly.

    Returns
    -------
    Neuron: the subsampled neuron
    """

    I = np.arange(neuron.n_soma, neuron.n_node)
    np.random.shuffle(I)
    selected_index = I[0:num - 1]
    selected_index = np.union1d([0], selected_index)
    selected_index = selected_index.astype(int)
    selected_index = np.unique(np.sort(selected_index))

    return neuron_with_selected_nodes(neuron, selected_index)


def regular_subsample(neuron):
    """
    Returning subsampled neuron with main nodes. i.e endpoints and branching
    nodes.

    Parameters
    ----------
    neuron: Neuron
        input neuron

    Returns
    -------
    Neuron: the subsampled neuron
    """
    # select all the main points
    selected_index = get_main_points(neuron)

    # Computing the parent id of the selected nodes
    neuron = neuron_with_selected_nodes(neuron, selected_index)
    return neuron



def prune_subsample(neuron, number):
    main_point = subsample_main_nodes()
    Nodes = main_point.nodes_list
    rm = (main_point.n_node - number)/2.
    for remove in range(int(rm)):
        b, m = find_sharpest_fork(Nodes)
        remove_pair_adjust_parent(Nodes, b)

    return Neuron(file_format = 'only list of nodes', input_file = Nodes)


def get_main_points(neuron):
    """
    Returning the index of branching points and end points.

    Parameters
    ----------
    neuron: Neuron
        input neuron

    Returns
    -------
    selected_index: array
        the list of main point; branching points and end points
    """
    (branch_index,) = np.where(neuron.branch_order[neuron.n_soma:] == 2)
    (endpoint_index,) = np.where(neuron.branch_order[neuron.n_soma:] == 0)
    selected_index = np.union1d(branch_index + neuron.n_soma,
                                endpoint_index + neuron.n_soma)
    selected_index = np.append(range(neuron.n_soma), selected_index)
    return selected_index

def neuron_with_selected_nodes(neuron, selected_index):
    """
    Giving back a new neuron made up with the selected_index nodes of self.
    if node A is parent (or grand parent) of node B in the original neuron,
    it is the same for the new neuron.

    Parameters
    ----------
    selected_index: numpy array
        the index of nodes from original neuron for making new neuron.

    Returns
    -------
    Neuron: the subsampled neuron.
    """
    parent = parent_id(neuron, selected_index)
    # making the list of nodes
    n_list = []
    for i in range(selected_index.shape[0]):
        n = Node()
        n.xyz = neuron.nodes_list[selected_index[i]].xyz
        n.r = neuron.nodes_list[selected_index[i]].r
        n.type = neuron.nodes_list[selected_index[i]].type
        n_list.append(n)
    # adjusting the childern and parents for the nodes.
    for i in np.arange(1, selected_index.shape[0]):
        j = parent[i]
        n_list[i].parent = n_list[j]
        n_list[j].add_child(n_list[i])
    return Neuron(file_format='only list of nodes', input_file=n_list)




def straigh_subsample(neuron, distance):
    """
    Subsampling a neuron from original neuron. It has all the main points of the original neuron,
    i.e endpoints or branching nodes, are not changed and meanwhile the distance of two consecutive nodes
    of subsample neuron is around the 'distance'.
    for each segment between two consecuative main points, a few nodes from the segment will be added to the selected node;
    it starts from the far main point, and goes on the segment toward the near main point. Then the first node which is
    going to add has the property that it is the farest node from begining on the segment such that its distance from begining is
    less than 'distance'. The next nodes will be selected similarly. this procesure repeat for all the segments.

    Parameters
    ----------
    distance: float
        the mean distance between pairs of consecuative nodes.

    Returns
    -------
    Neuron: the subsampled neuron
    """

    # Selecting the main points: branching nodes and end nodes
    selected_index = get_main_points(neuron)

    # for each segment between two consecuative main points, a few nodes from the segment will be added to the selected node.
    # These new nodes will be selected base on the fact that neural distance of two consecuative nodes is around 'distance'.
    # Specifically, it starts from the far main point, and goes on the segment toward the near main point. Then the first node which is
    # going to add has the property that it is the farest node from begining on the segment such that its distance from begining is
    # less than 'distance'. The next nodes will be selected similarly.

    for i in selected_index:
        upList = np.array([i], dtype = int)
        index = neuron.parent_index[i]
        dist = neuron.distance_from_parent[i]
        while(~np.any(selected_index == index)):
            upList = np.append(upList, index)
            index = neuron.parent_index[index]
            dist = np.append(dist, sum(neuron.distance_from_parent[upList]))
        dist = np.append(0, dist)
        (I,) = np.where(np.diff(np.floor(dist/distance))>0)
        I = upList[I]
        selected_index = np.append(selected_index, I)
    selected_index = np.unique(selected_index)
    neuron = neuron_with_selected_nodes(neuron, selected_index)
    return neuron


def straight_subsample_with_fixed_number(neuron, num):
    """
    Returning a straightened subsample neuron with fixed number of nodes.

    Parameters
    ----------
    num: int
        number of nodes on the subsampled neuron

    Returns
    -------
    distance: float
        the subsampling distance
    neuron: Neuron
        the subsampled neuron
    """
    l = sum(neuron.distance_from_parent)
    branch_number = len(np.where(neuron.branch_order[neuron.n_soma:] == 2))
    distance = l/(num - branch_number)
    neuron = straigh_subsample(neuron, distance)
    return neuron, distance

def parent_id2(neuron, selected_index):
    """
    Return the parent id of all the selected_index of the neurons.

    Parameters
    ----------
    selected_index: numpy array
        the index of nodes

    Returns
    -------
    parent_id: the index of parent of each element in selected_index in
    this array.
    """
    parent_id = np.array([], dtype=int)
    for i in selected_index:
        p = neuron.parent_index[i]
        while(~np.any(selected_index == p)):
            p = neuron.parent_index[p]
        (ind,) = np.where(selected_index == p)
        parent_id = np.append(parent_id, ind)
    return parent_id

def parent_id(neuron, selected_index):
    """
    Return the parent id of all the selected_index of the neurons.

    Parameters
    ----------
    selected_index: numpy array
        the index of nodes

    Returns
    -------
    parent_id: the index of parent of each element in selected_index in
    this array.
    """
    length = len(neuron.nodes_list)
    selected_length = len(selected_index)
    adjacency = np.zeros([length,length])
    adjacency[neuron.parent_index[1:], range(1,length)] = 1
    full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
    selected_full_adjacency = full_adjacency[np.ix_(selected_index,selected_index)]
    selected_adjacency = np.eye(selected_length) - np.linalg.inv(selected_full_adjacency)
    selected_parent_id = np.argmax(selected_adjacency, axis=0)
    return selected_parent_id

def prune(neuron,
          number_of_nodes,
          threshold):
    """
    Pruning the neuron. It removes all the segments that thier length is less
    than threshold unless the number of nodes becomes lower than lowest_number.
    In the former case, it removes the segments until the number of nodes is
    exactly the lowest_number.

    Parameters
    ----------
    neuron: Neuron
        input neuron.
    number_of_nodes: int
        the number of nodes for output neuron.

    Returns
    -------
    pruned_neuron: Neuron
        The pruned neuron.
    """
    n = len(neuron.nodes_list)
    for i in range(n - number_of_nodes):
        length, index = shortest_tips(neuron)
        if(length < threshold):
            neuron = remove_node(neuron, index)
        else:
            break
    neuron.set_distance_from_parent()
    return neuron

def remove_node(neuron, index):

    neuron.n_node -= 1
    node = neuron.nodes_list[index]
    parent_index = neuron.get_index_for_no_soma_node(node.parent)
    p = node.parent
    node.parent.remove_child(node)
    neuron.location = np.delete(neuron.location,index, axis = 1)
    neuron.nodes_list.remove(node)
    neuron.branch_order = np.delete(neuron.branch_order,index)
    new_parent_index = neuron.get_index_for_no_soma_node(p)
    neuron.branch_order[new_parent_index] -= 1
    neuron.parent_index = np.delete(neuron.parent_index,index)
    I = np.where(neuron.parent_index > index)
    neuron.parent_index[I] -= 1
    neuron.set_distance_from_parent()
    return neuron

def shortest_tips(neuron):
    """
    Returing the initial node of segment with the given end point.
    The idea is to go up from the tip.
    """
    (endpoint_index,) = np.where(neuron.branch_order[neuron.n_soma:] == 0)
    (branch_index,) = np.where(neuron.branch_order[neuron.n_soma:] == 2)
    selected_index = np.union1d(neuron.n_soma + endpoint_index,
                                neuron.n_soma + branch_index)
    selected_index = np.append(0, selected_index)
    par = parent_id(neuron, range(1,len(endpoint_index) + 1))
    dist = neuron.location[:, endpoint_index] - neuron.location[:, par]
    lenght = sum(dist**2,2)
    index = np.argmin(lenght)

    return np.sqrt(min(lenght)), endpoint_index[index] + neuron.n_soma


def straight_prune_subsample(neuron, number_of_nodes):
    """
    Subsampling a neuron with straightening and pruning. At the first step, it
    strighten the neuron with 200 nodes (if the number of nodes for the
    neuron is less than 200, it doesn't change it). Then the neuron is pruned
    with a twice the distance used for straightening. If the number of nodes
    is less than 'number_of_nodes' the algorithm stops otherwise it increases
    the previous distance by one number and does the same on the neuron.

    Parameters
    ----------
    neuron: Neuron
        input neuron
    number_of_nodes: int
        the number of nodes for the output neuron

    Returns
    -------
    sp_neuron: Neuron
        the subsample neuron after straightening and pruning.
    """
    if(neuron.n_node > 200):
        neuron, distance = straight_subsample_with_fixed_number(neuron, 200)
    sp_neuron = prune(neuron=neuron, number_of_nodes=number_of_nodes, threshold=2*distance)
    while(len(sp_neuron.nodes_list)>number_of_nodes):
        distance += 1
        sp_neuron = straigh_subsample(sp_neuron, distance)
        sp_neuron = prune(neuron=sp_neuron,
                                 number_of_nodes=number_of_nodes,
                                 threshold=2*distance)
    return sp_neuron

def mesoscale_subsample(neuron, number):
    main_point = subsample_main_nodes(neuron)
    Nodes = main_point.nodes_list
    rm = (main_point.n_node - number)/2.
    for remove in range(int(rm)):
        b, m = find_sharpest_fork(neuron, Nodes)
        remove_pair_adjust_parent(neuron, Nodes, b)
    neuron = Neuron(file_format = 'only list of nodes', input_file = Nodes)

    if(neuron.n_node > number):
        (I,) = np.where(neuron.branch_order == 0)
        neuron = remove_node(neuron, I[0])

    return neuron

def subsample_main_nodes(neuron):
    """
    subsamples a neuron with its main node only; i.e endpoints and branching nodes.

    Returns
    -------
    Neuron: the subsampled neuron
    """
    # select all the main points
    selected_index = get_main_points(neuron)

    # Computing the parent id of the selected nodes
    n = neuron_with_selected_nodes(neuron, selected_index)
    return n

def find_sharpest_fork(neuron, Nodes):
    """
    Looks at the all branching point in the Nodes list, selects those which both its children are end points and finds
    the closest pair of childern (the distance between children).
    Parameters
    ----------
    Nodes: list
    the list of Node

    Returns
    -------
    sharpest_pair: array
        the index of the pair of closest pair of childern
    distance: float
        Distance of the pair of children
    """
    pair_list = []
    Dis = np.array([])
    for n in Nodes:
        if n.parent is not None:
            if n.parent.parent is not None:
                a = n.parent.children
                if(isinstance(a, list)):
                    if(len(a)==2):
                        n1 = a[0]
                        n2 = a[1]
                        if(len(n1.children) == 0 and len(n2.children) == 0):
                            pair_list.append([n1 , n2])
                            dis = LA.norm(a[0].xyz - a[1].xyz,2)
                            Dis = np.append(Dis,dis)
    if(len(Dis)!= 0):
        (b,) = np.where(Dis == Dis.min())
        sharpest_pair = pair_list[b[0]]
        distance = Dis.min()
    else:
        sharpest_pair = [0,0]
        distance = 0.
    return sharpest_pair, distance

def find_sharpest_fork_general(neuron, Nodes):
    """
    Looks at the all branching point in the Nodes list, selects those which both its children are end points and finds
    the closest pair of childern (the distance between children).
    Parameters
    ----------
    Nodes: list
    the list of Node

    Returns
    -------
    sharpest_pair: array
        the index of the pair of closest pair of childern
    distance: float
        Distance of the pair of children
    """
    pair_list = []
    Dis = np.array([])
    for n in Nodes:
        if n.parent is not None:
            if n.parent.parent is not None:
                a = n.parent.children
                if(isinstance(a, list)):
                    if(len(a)==2):
                        n1 = a[0]
                        n2 = a[1]
                        pair_list.append([n1 , n2])
                        dis = LA.norm(a[0].xyz - a[1].xyz,2)
                        Dis = np.append(Dis,dis)
    if(len(Dis)!= 0):
        (b,) = np.where(Dis == Dis.min())
        sharpest_pair = pair_list[b[0]]
        distance = Dis.min()
    else:
        sharpest_pair = [0,0]
        distance = 0.
    return sharpest_pair, distance

def remove_pair_replace_node(neuron, Nodes, pair):
    """
    Removes the pair of nodes and replace it with a new node. the parent of new node is the parent of the pair of node,
    and its location and its radius are the mean of removed nodes.
    Parameters
    ----------
    Nodes: list
    the list of Nodes

    pair: array
    The index of pair of nodes. the nodes should be end points and have the same parent.

    Returns
    -------
    The new list of Nodes which the pair are removed and a mean node is replaced.
    """

    par = pair[0].parent
    loc = pair[0].xyz + pair[1].xyz
    loc = loc/2
    r = pair[0].r + pair[1].r
    r = r/2
    Nodes.remove(pair[1])
    Nodes.remove(pair[0])
    n = Node()
    n.xyz = loc
    n.r = r
    par.children = []
    par.add_child(n)
    n.parent = par
    Nodes.append(n)

def remove_pair_adjust_parent(neuron, Nodes, pair):
    """
    Removes the pair of nodes and adjust its parent. the location of the parent is the mean of the locaton of two nodes.

    Parameters
    ----------
    Nodes: list
    the list of Nodes

    pair: array
    The index of pair of nodes. the nodes should be end points and have the same parent.

    Returns
    -------
    The new list of Nodes which the pair are removed their parent is adjusted.
    """

    par = pair[0].parent
    loc = pair[0].xyz + pair[1].xyz
    loc = loc/2
    Nodes.remove(pair[1])
    Nodes.remove(pair[0])
    par.xyz = loc
    par.children = []


def parent_id_for_extract(original_parent_id, selected_index):
    """
    Return the parent id of all the selected_index of the neurons.

    Parameters
    ----------
    selected_index: numpy array
        the index of nodes

    Returns
    -------
    parent_id: the index of parent of each element in selected_index in
    this array.
    """
    length = len(original_parent_id)
    selected_length = len(selected_index)
    adjacency = np.zeros([length, length])
    adjacency[original_parent_id[1:]-1, range(1, length)] = 1
    full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
    selected_full_adjacency = full_adjacency[np.ix_(selected_index, selected_index)]
    selected_adjacency = np.eye(selected_length) - np.linalg.inv(selected_full_adjacency)
    selected_parent_id = np.argmax(selected_adjacency, axis=0)
    return selected_parent_id


def parent_id_for_extract2(original_parent_id, selected_index):
    parent_id = np.array([], dtype=int)
    for i in selected_index[1:]:
        p = original_parent_id[i]
        while(~np.any(selected_index == p-1)):
            p = original_parent_id[p-1]
        (ind,) = np.where(selected_index == p-1)
        parent_id = np.append(parent_id, ind)
    parent_id = np.append(-1, parent_id)
    return parent_id

def extract_main_neuron_from_swc(matrix, num = 300):
    a, b = np.unique(matrix[:,6],return_counts=True)
    (I,) = np.where(b==2)
    branch_point = a[I]
    end_point = np.setxor1d(np.arange(0,matrix.shape[0]), matrix[4:,6])
    I = np.union1d(branch_point, end_point)
    leng = matrix.shape[0]
    lengi = len(I)
    I = np.union1d(I, np.arange(0,n, int(n/(leng - lengi - 1))))
    random_point = np.setxor1d(np.arange(3,matrix.shape[0]), I)
    I = np.append(I, random_point[-(num - len(I)):-1])
    I = np.sort(I)
    I = np.array(I,dtype=int)
    I[0] = 0
    K = matrix[:,6]
    K = np.array(K,dtype=int)
    J = parent_id_for_extract2(K, I)

    J = J + 1
    J[0] = -1
    n = len(J)

    I = I - 1
    I = I[1:]
    I = np.append(I,matrix.shape[0]-1)
    new_matrix = np.zeros([n, 7])
    new_matrix[:,0] = np.arange(0,n)
    new_matrix[:,1] = matrix[I,1]
    new_matrix[:,2] = matrix[I,2]
    new_matrix[:,3] = matrix[I,3]
    new_matrix[:,4] = matrix[I,4]
    new_matrix[:,5] = matrix[I,5]
    new_matrix[:,6] = J
    neuron = Neuron(file_format='Matrix of swc', input_file=new_matrix)
    return neuron
