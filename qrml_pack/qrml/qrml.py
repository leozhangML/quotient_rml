import numpy                as np
from sklearn.neighbors      import KDTree
from sklearn.decomposition  import PCA
from scipy.sparse           import csr_matrix
from scipy.sparse.csgraph   import dijkstra
from scipy.spatial          import distance_matrix
from scipy.stats            import mode
from scipy.spatial          import Delaunay
import matplotlib.pyplot    as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import gurobipy             as gp
from gurobipy               import GRB
from kneed                  import KneeLocator
import warnings

warnings.filterwarnings("ignore")

# dimension estimation
def local_pca_elbow(pointcloud, S1):
    """
    Applies PCA to local pointclouds and recovers its
    local dimension by finding the elbow point in the 
    function of recovered variances.

    Parameters
    ----------
    pointcloud : (n_samples, n_features) np.array
        Subarray of pointcloud data.
    S1 : positive float
        Parameter for KneeLocator.

    Returns
    -------
    dim : int
        Estimated dimension of local pointcloud.
    """

    if len(pointcloud) == 1:
        return 1

    pca = PCA()
    _ = pca.fit(pointcloud)
    vs = pca.explained_variance_ratio_

    kneedle = KneeLocator([i for i in range(len(vs))], vs, S=S1, curve='convex', direction='decreasing')
    elbow = kneedle.elbow
    dim = elbow + 1 if elbow!=None else 0

    return dim

# boundary functions
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation

    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """

    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def find_tear_points(S, a):
    """
    Computes the set of points ("tear points") in S.pointcloud 
    where the ratio of mean length of the projected edges 
    (to a given point) and mean length of the original edges 
    is greater than a. 

    Parameters
    ----------
    S : Simplex object
        Needs to have coords computed.
    a : positive float

    Returns
    -------
    tear_points : list
        List of the indexes of points in S.pointcloud
        which are classed as "tear points".
    """

    dist_matrix = dijkstra(S.edge_matrix)
    tear_points = []

    for i, edge in enumerate(S.edges):
        manifold_dist = np.mean(dist_matrix[i, edge])
        proj_dist = np.mean(np.linalg.norm(S.coords[edge] - S.coords[i], axis=1))

        if proj_dist / manifold_dist > a:
            tear_points.append(i)

    return tear_points

def find_orientation(edges, **kwargs):
    """
    From the output of alpha_shape, we order the edges
    to give an orientation - assumes edges gives a 1-cycle.

    Parameters
    ----------
    edges : (num_edges,) set 
        Set of (i,j) pairs representing edges of the 
        alpha-shape. (i,j) are the indices in the points array.
    
    Returns
    -------
    orientation : (num_edges, 2) np.array 
        The ith row gives the indexes (j, k) of the connected 
        points of the ith edge in the orientated boundary.
    """

    orientation = []
    n_edges = np.asarray(list(edges))
    point = n_edges[0, 0]

    for _ in range(len(n_edges)):
        edge_idx = np.where(n_edges[:, 0]==point)[0][0]  # next point from point
        point = n_edges[edge_idx][1]
        orientation.append(n_edges[edge_idx])
    
    return np.asarray(orientation)

def add_boundary(S, orientation, ax, three_d=False, alpha0=0.5, cmap='cool', loop=True, **kwargs):
    """
    Applies the orientated edge given by orientation onto ax
    with the colouring given by cmap. The orientation goes 
    from blue to pink if cmap=='cool'.

    Adapted from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html.

    Parameters
    ----------
    S : Simplex object
        Needs to have coords computed.
    orientation : (num_edges, 2) np.array 
        The ith row gives the indexes (j, k) of the connected 
        points of the ith edge in the orientated edge.
    ax : matplotlib.axes.Axes object
        Axes to plot the orientated edge on.
    three_d : bool
        True if the axes we use is for a 3D plot.
    alpha0 : float
        Value for alpha for plotting.
    cmap : colourmap
        Colourmap to use to plot the edge.
    loop : bool
        True if the edge to be plotted is a 1-cycle.

    Returns
    -------
    ax : matplotlib.axes.Axes object
        Axes with the plotted orientated edge applied.
    """

    if three_d:
        if loop:
            xy = S.pointcloud[np.concatenate([orientation[:, 0], np.array([orientation[0, 0]])])]
        else:
             xy = S.pointcloud[orientation[:, 0]]
        points = xy.reshape(-1, 1, 3)
    else:
        if loop:
            xy = S.coords[np.concatenate([orientation[:, 0], np.array([orientation[0, 0]])])]
        else:
            xy = S.coords[orientation[:, 0]] 
        points = xy.reshape(-1, 1, 2) 

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    grading = np.linspace(0, 0.7, len(xy)-1)
    norm = plt.Normalize(grading.min(), grading.max())

    if three_d:
        lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha0)
    else:
        lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha0)
        
    lc.set_array(grading)
    lc.set_linewidth(10)
    line = ax.add_collection(lc)

    return ax

# quotient identification functions
def orientation_dist(current_idx, short_node_idx, n):
    """
    Computes the distance (how many steps away) between 
    points in the orientated boundary of size n.

    Parameters
    ----------
    current_idx : int
        Index of point in orientation.
    short_node_idx : int
        Index of point in orientation
    n : int 
        length of orientation.
    
    Returns
    -------
    out : int
        How many steps away the input indexes are.
    """

    return min((current_idx-short_node_idx)%n, (-current_idx+short_node_idx)%n)

def clean_boundary(S, tol, orientation, boundary_edges):
    """
    We "clean" our orientated boundary by closing up
    loops in the boundary - given by points in the boundary
    connecting (by the 1-skeleton) to points other than its neighbours 
    in the boundary.

    For a given point in the boundary, if it connects to another point in
    the boundary at most tol points ahead (in the orientated boundary),
    then we close up this short-circuited loop by discarding the intermediate 
    points.

    Parameters
    ----------
    S : Simplex object
    tol : int >= 2
        Length of loops to close up.
    orientation : (num_edges, 2) np.array 
        The ith row gives the indexes (j, k) of the connected 
        points of the ith edge in the orientated boundary.
    boundary_edges : (num_edges,) set 
        Set of (i,j) pairs representing edges of the 
        alpha-shape. (i,j) are the indices in the points array.

    Returns
    -------
    included : (num_edges,) bool np.array
        The ith entry is True if the ith point in the orientated
        boundary is included in the "clean" boundary.
    """

    if len(np.unique(orientation[:, 0])) != len(boundary_edges):  # should be equal for 1-cycle
        print('Boundary is not a well-defined 1-cycle - try a different alpha value!')
        return -1

    orientation = orientation[:, 0]  # in terms of the indicies of S.coords in boundary
    n = len(orientation)

    included = np.full(n, True)
    order = {node: count for count, node in enumerate(orientation)}  # (node, place in orientation)

    # as the boundary is dynamically changing, we loop until no more changes are possible
    keep_looping = True
    while keep_looping:
        keep_looping = False

        # loop from the start of our orientation until the end
        for node_idx in range(n):
            if included[node_idx] == False:
                continue
            else:
                visible_nodes = []

                # finds the visible nodes from node_idx
                for i in range(1, n):
                    if included[(node_idx+i)%n] == False:
                        continue
                    else:
                        visible_nodes.append(orientation[(node_idx+i)%n])
                        if len(visible_nodes) == tol:
                            break

                # handles connections
                test_nodes = visible_nodes.copy()
                test_nodes.pop(0)  # we ignore the first node connected to node_idx
                test_nodes = set(test_nodes)

                local_edges = set(S.edges[orientation[node_idx]])
                intersection = test_nodes.intersection(local_edges)

                if len(intersection) != 0:
                    intersection = list(intersection)
                    furthest_idx = np.argmax([(order[i]-node_idx)%n for i in intersection])
                    furthest_point = intersection[furthest_idx]  # a_i
                    # visible nodes is already ordered by orientation from node_idx
                    deleted_points = [order[i] for i in visible_nodes[:visible_nodes.index(furthest_point)]]
                    included[deleted_points] = False
                    keep_looping = True

    return included

def intersection(S, clean_orientation, connection_tol):
    """
    For each point in clean_orientation, we compute the set of 
    points in clean_orientation which connect to it; connection_tol
    specifies how far these points must be to count as a connection.

    If this set is non-empty, we record the index (in clean_orientation) 
    of such a point.

    Parameters:
    -----------
    S : Simplex Object
    clean_orientation : (num_edges,) np.array 
        The array lists the indexes of boundary points in order
        (in terms of S.coords).
    connection_tol : non-negative int
        Specifies how far connected points must be to count 
        as a short-circuit connection.

    Returns:
    --------
    short_connections : list
        The ith element is the list of short-circuit connections 
        (in terms of S.coords indexes) of the ith node with 
        local points - as specified by connection_tol - removed.
    short_idxs : list
        List of the indexes of short-circuit points
    """

    n = len(clean_orientation)
    set_clean_orientation = set(clean_orientation)
    short_connections = []
    short_idxs = []

    for i, node in enumerate(clean_orientation):
        local_edges = set(S.edges[node])
        neighbour_edges = set(clean_orientation[(i+j)%n] for j in range(-connection_tol, connection_tol+1)) 
        local_edges -= neighbour_edges
        local_edges = local_edges.intersection(set_clean_orientation)
        if len(local_edges) != 0:
            short_idxs.append(i)
        local_edges = list(local_edges)
        short_connections.append(local_edges)

    return short_connections, short_idxs

def identify_edges(n, short_idxs, quotient_tol):
    """
    Identify short-circuit and non-short-circuit edges in clean_orientation.

    Non-short-circuit edges are collections of contiguous points which are
    classed as not glued to any other edge in the boundary. 

    Short-circuit edges (or short-edges) are the same but are classed as glued 
    to some other (short-circuit) edge.

    Parameters
    ----------
    n : int
        Length of clean_orientation.
    short_idxs : list 
        List of indexes (in clean_orientation) where short-circuits occur.
    quotient_tol : positive int
        Sets how many non-short-circuit points between short-circuit points
        we allow before forming a non-connected edge.
    
    Returns
    -------
    short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        short-circuit edges.
    non_short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        non-connected edges.
    """

    short_edges = []
    non_short_edges = []

    base = short_idxs[0]
    current_idx = short_idxs[0]
    short = False
    for short_idx in (short_idxs[1:]+[short_idxs[0]+n]):  # to allow to loop back
        diff = short_idx - current_idx
        if diff <= quotient_tol:  # quotient_tol is how many points in front of a short-circuit node before another such point do we allow before forming a non-short-circuit edge
            short = True
        else:
            if short == True:
                short_edges.append([i%n for i in range(base, current_idx+1)])  # [base, current_idx] is short
                non_short_edges.append([i%n for i in range(current_idx, short_idx+1)])  # [current_idx, short_idx] is not short
            else:
                non_short_edges.append([i%n for i in range(current_idx, short_idx+1)])  # [current_idx, short_idx] is not short
            base = short_idx
            short = False
        current_idx = short_idx

    # take care of case with all True - i.e. only one edge (short-circuit)
    if len(non_short_edges) == 0:
        short_edges.append([i for i in range(n)])
    # takes care of the loop in the orientation of the boundary
    else:
        try:
            first_short = True if short_idxs[0] in short_edges[0] else False  # True if the first short-circuit point lies in a short-circuit section
        except:
            first_short = False  # if short_idxs is empty - i.e. only non-short-circuit edges
        if short and first_short:
            short_edges[0] = [i%n for i in range(base, short_idxs[0]+n)] + short_edges[0]
        elif short and not first_short:
            short_edges.append([i%n for i in range(base, short_idxs[0]+n+1)])
    
    return short_edges, non_short_edges

def refine_edges(short_edges, tol1, clean_orientation, short_connections):
    """
    Refines the short-edges (split into multiple short-edges) to take 
    care of self intersections in a short-edge - i.e. RP2 and torus.

    Parameters
    ----------
    short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        short-circuit edges.
    tol1 : positive int
        The number of points that short-circuits points must at
        least be seperated by to count as well-defined splitting points.
    clean_orientation: 1-D np.array
        Indexes (in S.coords) of boundary points.
    short_connections : list of lists
        ith element is the list of short-circuit connections of the ith
        node with the predeccesor and succesor (in clean_orientation)
        removed.

    Returns
    -------
    refined_short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        (refined) short-circuit edges.
    """

    refined_short_edges = []
    clean_order = {node: idx for idx, node in enumerate(clean_orientation)}
    n = len(clean_orientation)

    for short_edge in short_edges:
        refine = False
        max_self_connections_idxs = None  # to be in terms of the indexes of short_edge
        max_self_connections_node = None
        max_self_connections = 0
        set_short_edge = set(clean_orientation[short_edge])

        for idx, node in enumerate(clean_orientation[short_edge]):
            node_idx = clean_order[node]
            self_connections = set_short_edge.intersection(set(short_connections[node_idx]))  # set of self intersections in terms of clean_orientation
            if len(self_connections) <= max_self_connections:
                continue
            else:
                self_connections_idxs = np.asarray([np.where(clean_orientation==i)[0][0] for i in self_connections])
                self_connections_idxs = self_connections_idxs[np.argsort(self_connections_idxs)]
                chosen_points = []

                current_idx = node_idx
                for short_node_idx in self_connections_idxs:
                    if orientation_dist(node_idx, short_node_idx, n)>tol1 and orientation_dist(current_idx, short_node_idx, n)>tol1:  # could have case of short-circuit connections which are very close - avoid this as it doesn't represent seperate edges
                        chosen_points.append(np.where(clean_orientation[short_edge]==clean_orientation[short_node_idx])[0][0])  # taken in terms of short_edge indexes
                    current_idx = short_node_idx
                if len(chosen_points) > max_self_connections:  # pick the node with the most (safe) short-circuit connections to its own edge
                    refine = True
                    max_self_connections_idxs = chosen_points
                    max_self_connections_node = idx
                    max_self_connections = len(chosen_points)
        if refine:
            splitting_points = np.sort(max_self_connections_idxs+[max_self_connections_node])
            if len(short_edge) == n:  # if this short-edge is the full loop 
                splitting_points = np.append(splitting_points, splitting_points[0]+n)
            for idx in range(len(splitting_points)-1):
                slice_range = [i%n for i in range(splitting_points[idx], splitting_points[idx+1]+1)]
                refined_short_edges.append(list(np.asarray(short_edge)[slice_range]))
        else:
            refined_short_edges.append(short_edge)

    return refined_short_edges

def connect_edges(short_edges, clean_orientation, short_connections):
    """
    Find which short-edges are identified together.

    Parameters
    ----------
    short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        short-circuit edges.
    clean_orientation: 1-D np.array
        Indexes (in S.coords) of boundary points.
    short_connections : list of lists
        ith element is the list of short-circuit connections of the ith
        node with the predeccesor and succesor (in clean_orientation)
        removed.
    
    Returns
    -------
    glued_edges : list of lists
        Consist of elements [i, j] where i, j are indexes 
        in short_edges representing "glued" edges.
    """
    edge_connections = []
    clean_order = {node: idx for idx, node in enumerate(clean_orientation)}

    for short_edge in short_edges:
        short_edge_connections = set()
        for node in clean_orientation[short_edge]:
            short_edge_connections.update(short_connections[clean_order[node]])
        edge_connections.append(short_edge_connections)
    glued_edges = set()

    for idx, edge_connection in enumerate(edge_connections):
        short_count = []
        for short_edge in short_edges:
            count = len(edge_connection.intersection(set(clean_orientation[short_edge])))
            short_count.append(count)
        connected_edge_idx = np.argmax(short_count)
        glued_edges.add(frozenset([idx, connected_edge_idx]))  # associates the two together in terms of index in short_edges

    glued_edges = [list(i) for i in glued_edges]

    """
    refined_glued_edges = []

    for pair in glued_edges:  # ignores single gluings if some other non-single relevant gluing exists
        allow = True
        if len(pair) == 1:
            for pair1 in glued_edges:
                if pair[0] in pair1 and len(pair1) != 1:
                    allow = False
        if allow:
            refined_glued_edges.append(pair)

    glued_edges = refined_glued_edges
    """

    return glued_edges

def find_connection_order(node, compared_short_edge, short_connections, clean_order):
    """
    Computes the maximum and minimum index (in the compared_short_edge) of the 
    points which connect to node.

    Parameters
    ----------
    node : positive int
        Element of clean_orientation
    compared_short_edge : 1-D np.array 
        Sub-array of clean_orientation representing a short-circuit edge
    short_connections : list of lists
        ith element is the list of short-circuit connections of the ith
        node with the predeccesor and succesor (in clean_orientation)
        removed.
    clean_order : dict
        Keys are elements of clean_orientation with value equal to their index
        in the array.

    Returns
    -------
    None/(maximum, minimum)
    """

    set_compared_short_edge = set(compared_short_edge)
    compared_connections = set(short_connections[clean_order[node]]).intersection(set_compared_short_edge)
    if len(compared_connections) == 0:
        return None
    else:
        orders = [np.where(compared_short_edge==i)[0][0] for i in compared_connections]  # finds the positions in compared_short_edge of the connections to node ! this is ordered by orientation 
        maximum = max(orders)
        minimum = min(orders)
        return maximum, minimum

def gluing_orientation(glued_edges, short_edges, clean_orientation, short_connections):
    """
    Finds the orientation of each glued pair.

    Parameters
    ----------
    glued_edges : list of lists
        Consist of elements [i, j] where i, j are indexes 
        in short_edges representing "glued" edges.
    short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        short-circuit edges.
    clean_orientation: 1-D np.array
        Indexes (in S.coords) of boundary points.
    short_connections : list of lists
        ith element is the list of short-circuit connections of the ith
        node with the predeccesor and succesor (in clean_orientation)
        removed.

    Returns
    -------
    same_orientation : bool list
        ith element is True if ith pair in glued_edges have the same
        orientation. False, otherwise.
    """

    same_orientation = []  # the ith element is True if the glued edges in glued_edges[i] have the same orientation and False otherwise
    clean_order = {node: idx for idx, node in enumerate(clean_orientation)}
    for idxs in glued_edges:
        if len(idxs) == 1:
            print('Short-circuit edges not properly seperated! Try new parameters!')  # TODO : try going to second most connected edge? Should give correct orientation for torus?
            return -1
        else:
            idx1, idx2 = idxs
            short_edge = clean_orientation[short_edges[idx1]]  # ordered by orientation and in terms of clean_orientation
            compared_short_edge = clean_orientation[short_edges[idx2]]

            same_orientation_count = 0
            reverse_orientation_count = 0
            start = True
            current_max = None
            current_min = None

            for short_node in short_edge:
                short_order = find_connection_order(short_node, compared_short_edge, short_connections, clean_order)
                if short_order == None:
                    continue
                else:
                    max_order, min_order = short_order  # tries to account for noise by seeing what the biggest consistent gluing is for each orientation
                    if start != True:
                        if max_order >= current_min:
                            reverse_orientation_count += 1
                        if min_order <= current_max:
                            same_orientation_count += 1
                    else:
                        start = False
                    current_max = max_order
                    current_min = min_order

            if reverse_orientation_count > same_orientation_count:
                same_orientation.append(False)
            else:
                same_orientation.append(True)

    return same_orientation

def assemble_quotient(glued_edges, same_orientation):
    """
    Assigns how to colour and plot our glued edges (such that we 
    can handle the case when more than three edges are glued 
    together etc.). Checks for compatible gluing.

    Parameters
    ----------
    glued_edges : list of lists
        Consist of elements [i, j] where i, j are indexes 
        in short_edges representing "glued" edges.
    same_orientation : bool list
        ith element is True if ith pair in glued_edges have the same
        orientation. False, otherwise.
    
    Returns
    -------
    orientation_dict : dict
        Keys are indexes for short_edges with value as True if the indexed
        edge should be plotted with the same orientation as induced from 
        clean_orientation. False, otherwise.
    colour_dict : dict
        Keys are indexes for short_edges with int values. Elements with the
        same values are plotted with the same colour - i.e. glued together.
    """

    colour_count = 0
    orientation_dict = {}  # orientation_dict[i] is True if short_edges[i] has the same orientation as from clean_orientation
    colour_dict = {}  # this allows us to assign edges the same colour if glued when more than two edges are glued together

    for idx, (i, j) in enumerate(glued_edges):

        if i not in orientation_dict.keys() and j not in orientation_dict.keys():  # if (i, j) are a new glued pair: i keeps orientation and j flips. if i is already glued and (i, k) is a new glued edge then if same_orientation then k is the opposite of i otherwise the same
            if same_orientation[idx]:
                orientation_dict[i] = True
                colour_dict[i] = colour_count
                orientation_dict[j] = False
                colour_dict[j] = colour_count
            else:
                orientation_dict[i] = True
                colour_dict[i] = colour_count
                orientation_dict[j] = True
                colour_dict[j] = colour_count
            colour_count += 1

        elif i in orientation_dict.keys() and j in orientation_dict.keys():  # check compatible
            if same_orientation[idx]:
                if orientation_dict[i] == orientation_dict[j]:
                    print('Incompatible gluing! Try different parameters.')
                    return -1
            else:
                if orientation_dict[i] != orientation_dict[j]:
                    print('Incompatible gluing! Try different parameters.')
                    return -1
            colour_dict[j] = colour_dict[i]  # if compatible then switch all colours to match
            for glued_edge in glued_edges:
                if j in glued_edge:
                    k = glued_edge[0] if glued_edge[0]!=j else glued_edge[1]
                    colour_dict[k] = colour_dict[j]

        else:
            i1 = i if orientation_dict.get(i) != None else j  # if edge in the pair has already been processed but the other has not
            j1 = i if orientation_dict.get(i) == None else j  
            if same_orientation[idx]:
                orientation_dict[j1] = not orientation_dict[i1]
            else:
                orientation_dict[j1] = orientation_dict[i1]
            colour_dict[j1] = colour_dict[i1]

    return orientation_dict, colour_dict

def convert_orientation(orientation):
    """
    Takes a 1-D orientation (full 1-cycle) such 
    clean_orientation to 2-D np.array (like orientation) for plotting.

    Parameters
    -------
    orientation : (num_edges, 2) np.array 
        The ith row gives the indexes (j, k) of the connected 
        points of the ith edge in the orientated boundary.
    """

    return np.column_stack([orientation[:], np.concatenate([orientation[1:], np.array([orientation[0]])])])

# quotient diagnostics
def find_short_and_refined(S, alpha, tol, quotient_tol, tol1, connection_tol=5):
    """
    Computes the intial short-edges, non-short-edges
    and the refined short-edges.

    Parameters
    ----------
    alpha : float
        Value for alpha shapes.
    tol : int >= 2
        Length of loops to close up with clean_boundary.
    quotient_tol : int
        Number of points between short-circuit points
        required (at least) to define a non-short-circuit 
        between them.
    tol1 : int
        Minimum distance required between splitting points 
        when refining edges.
    connection_tol : int
        Specifies how far connected points must be to count 
        as a short-circuit connection.

    Returns
    -------
    short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        short-circuit edges.
    refined_short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        (refined) short-circuit edges.
    non_short_edges : list of lists
        Each list gives the indexes in clean_orientation of the 
        non-connected edges.
    orientation : (num_edges, 2) np.array 
        The ith row gives the indexes (j, k) of the connected 
        points of the ith edge in the orientated boundary.
    clean_orientation : (num_edges,) np.array 
        The array lists the indexes of boundary points in order
        (in terms of S.coords).
    short_connections : list
        The ith element is the list of short-circuit connections 
        (in terms of S.coords indexes) of the ith node with 
        local points - as specified by connection_tol - removed.
    """

    if np.all(S.coords==None):
        print('No projection found! Try normal_coords.')
        return None
    elif S.dim != 2:
        print("Dimension of projection is not two - cannot compute quotient!")
        return -1

    # set up boundary with orientation
    edges = alpha_shape(S.coords, alpha=alpha)
    orientation = find_orientation(edges)

    # clean up boundary
    included = clean_boundary(S, tol, orientation, edges)
    if np.sum(included) == 2:  # if all points are identified  
        return None, None, None, orientation, orientation[:, 0], None
    elif np.all(included==-1):  # if orientation is not 1-cycle
        return -1
    clean_orientation = orientation[:, 0][included]  # 1-D array
    n = len(clean_orientation)

    # set up connective structure of the "cleaned" orientation
    short_connections, short_idxs = intersection(S, clean_orientation, connection_tol)
    m = len(short_idxs)
    if m == 0:  # if only one non-short-circuit 1-cycle 
        return None, None, [[i for i in range(n)]], orientation, clean_orientation, short_connections

    # identify and refine edges
    short_edges, non_short_edges = identify_edges(n, short_idxs, quotient_tol)
    if len(short_edges) == 0:
        return None, None, non_short_edges, orientation, clean_orientation, short_connections
    refined_short_edges = refine_edges(short_edges, tol1, clean_orientation, short_connections)

    return short_edges, refined_short_edges, non_short_edges, orientation, clean_orientation, short_connections

def plot_edges(S, c, edge_info, alpha0=0.8):
    """
    Plots the intial intial short-edges, non-short-edges
    as well as the refined short-edges and non-short-edges
    for comparison.

    Short-edges are plotted with solid lines and non-short-edges
    are plotted with dotted lines. We also show the connection
    of boundary points to boundary points.

    Parameters
    ----------
    c : colour map
    edge_info : tuple
        Output of find_short_and_refined.
    alpha0 : float
        Alpha value (for line plotting).
    """

    if edge_info == -1:
        return None

    short_edges, refined_short_edges, non_short_edges, orientation, clean_orientation, short_connections = edge_info
    clean_orientation_2d = convert_orientation(clean_orientation)

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text('short_edges and non_short_edges')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text('refined_short_edges and non_short_edges')
    plt.axis('equal')

    # setting plotting colours
    colours = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'cyan']
    orientation_cmaps = ['cool', 'Purples', 'Blues', 'Greens', 'Oranges',
                         'YlOrBr', 'YlOrRd', 'OrRd', 'Greys', 'PuRd', 'RdPu', 
                         'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    ax1.scatter(S.coords[:, 0], S.coords[:, 1], c=c)  # for short_edges and non_short_edges
    ax2.scatter(S.coords[:, 0], S.coords[:, 1], c=c)  # for refined_short_edges and non_short_edges

    # show how short-circuit nodes are connected to boundary
    for idx, short_connection in enumerate(short_connections):
        if len(short_connection) != 0:
            ax1.scatter(S.coords[clean_orientation_2d[:, 0][idx], 0], S.coords[clean_orientation_2d[:, 0][idx], 1], c='r', marker='o')
            ax2.scatter(S.coords[clean_orientation_2d[:, 0][idx], 0], S.coords[clean_orientation_2d[:, 0][idx], 1], c='r', marker='o')
            for j in short_connection:
                ax1.plot([S.coords[clean_orientation_2d[:, 0][idx], 0], S.coords[j][0]], [S.coords[clean_orientation_2d[:, 0][idx], 1], S.coords[j][1]], c='black', alpha=0.5)
                ax2.plot([S.coords[clean_orientation_2d[:, 0][idx], 0], S.coords[j][0]], [S.coords[clean_orientation_2d[:, 0][idx], 1], S.coords[j][1]], c='black', alpha=0.5)

    # handles looping of boundary orientation if no short-edges exist 
    no_short_edges = False
    if short_edges == None:
        no_short_edges = True
    else:
        if len(short_edges) == 0:
            no_short_edges = True

    # handles looping of boundary orientation if no non-short-edges exist 
    no_non_short_edges = False
    if non_short_edges == None:
        no_non_short_edges = True
    else:
        if len(non_short_edges) == 0:
            no_non_short_edges = True

    # handles non_short_edges for ax1 and ax2
    if non_short_edges != None:
        if len(non_short_edges) == 1 and no_short_edges:  # if single non-short-circuit 1-cycle 
            ax1 = add_boundary(S, clean_orientation_2d, ax1, three_d=False, alpha0=alpha0)
            ax2 = add_boundary(S, clean_orientation_2d, ax2, three_d=False, alpha0=alpha0)
        else:
            for idx, non_short_edge in enumerate(non_short_edges):
                edge = clean_orientation_2d[non_short_edge]
                l = len(edge)
                for i, j in edge:
                    ax1.plot([S.coords[i][0], S.coords[j][0]],[S.coords[i][1], S.coords[j][1]], color=colours[idx], linewidth=10, alpha=alpha0, linestyle='dotted')
                    ax2.plot([S.coords[i][0], S.coords[j][0]],[S.coords[i][1], S.coords[j][1]], color=colours[idx], linewidth=10, alpha=alpha0, linestyle='dotted')

    # handles sort_edges for ax1
    if short_edges != None:
        if len(short_edges) == 1 and no_non_short_edges:
            loop = True
        else:
            loop = False
        for idx, short_edge in enumerate(short_edges):
            short_edge_clean_orientation = clean_orientation_2d[short_edge]
            ax1 = add_boundary(S, short_edge_clean_orientation, ax1, three_d=False, alpha0=alpha0, cmap=orientation_cmaps[idx], loop=loop)

    # handles refined_short_edges for ax2
    if refined_short_edges != None:
        if len(refined_short_edges) == 1 and no_non_short_edges:
            loop = True
        else:
            loop = False
        for idx, refined_short_edge in enumerate(refined_short_edges):
            refined_short_edge_clean_orientation = clean_orientation_2d[refined_short_edge]
            ax2 = add_boundary(S, refined_short_edge_clean_orientation, ax2, three_d=False, alpha0=alpha0, cmap=orientation_cmaps[idx], loop=loop)

    # if all points are associated 
    if non_short_edges == None and short_edges == None:
        for i, j in clean_orientation_2d:
            ax1.plot([S.coords[i][0], S.coords[j][0]],[S.coords[i][1], S.coords[j][1]], color='black', linewidth=10, alpha=alpha0)
            ax2.plot([S.coords[i][0], S.coords[j][0]],[S.coords[i][1], S.coords[j][1]], color='black', linewidth=10, alpha=alpha0)

    plt.show()


class Simplex:
    """
    Class for computing projections and
    quotients in qrml.
    """

    def __init__(self):
        """
        Attributes
        ----------
        pointcloud : (n_samples, n_features) np.array
            The pointcloud data from which we build our simplex. 
        edges : (n_samples,) list containing 1-D np.array
            The ith entry contains the indexes of the 'safe' points which 
            connect to the ith point.
        edge_matrix : (n_samples, n_samples) np.array / csr_matrix
            Gives the structure of our edge connection. The (i, j)
            entry gives the length of the edge between the ith and jth
            points.
        dim : int
            The estimated dimension of our pointcloud.
        coords : (n_samples, self.dim) np.array
            Projected coordinates from the "naive" algorithm.
        p_idx : int
            Index of the point in self.pointcloud that we take as the 
            base point in our projection.
        """

        self.pointcloud = None
        self.edges = []
        self.edge_matrix = None
        self.dim = None
        self.coords = None
        self.p_idx = None

    def find_visible_edge(self, idx, ind, dist):
        """
        Computes a list of the indexes of points visible from 
        the 'idx' point, and their distances from this point 
        (in ascending length).

        Parameters
        ----------
        idx : int
            Index of a point.
        ind : (k,) np.array
            Indexes of points connected to the idx point by KNN.
        dist : (k,) np.array
            Array of edge lengths from KNN.

        Returns
        -------
        visible_ind : (n_visible_edges,) np.array
            Indexes of visible edges (self.pointcloud).
        visible_dist : (n_visible_edges,) np.array
            Lengths of visible edges.
        """

        point = self.pointcloud[idx]
        # List of indexes for visible points from the 'idx' point
        # where the indexes are for 'ind' (not self.pointcloud)
        visible_points_idx = [] 
        visible = True

        for y_count, idy in enumerate(ind):
            y = self.pointcloud[idy]
            for idz in ind:
                if idz != idy:
                    z = self.pointcloud[idz]
                    cos_angle = np.dot(point - z, y - z)  
                    if cos_angle < 0:
                        visible = False
                        break
            if visible == True:
                visible_points_idx.append(y_count)
            visible = True

        visible_dist = dist[visible_points_idx]  
        visible_ind = ind[visible_points_idx] 

        return visible_ind, visible_dist 

    def find_safe_edges(self, idx, ind, dist, threshold_var, edge_sen):
        """
        Computes the list of safe edges of points from visible edges 
        which connect to the 'idx' point to self.edge_matrix.

        Parameters
        ----------
        idx : int
            Index of a point.
        ind : (k,) np.array
            Indexes of visible points connected to the idx point.
        dist : (k,) np.array
            Array of edge lengths.
        threshold_var : [0,1] float
            The threshold to estimate the local intrinsic dimension by PCA.
        edge_sen : positive float
            The sensitivity with which we choose safe edges.
        """

        point = self.pointcloud[idx]
        edges = self.pointcloud[ind] - point  # ascending by length
        threshold_edge = edge_sen * np.mean(dist)

        for j in range(2, len(edges)+1):  # need len != 1 
            pca = PCA()
            pca.fit_transform(edges[:j])
            var = pca.explained_variance_ratio_

            if j == 2:
                dim0 = dim1 = np.sum(var >= threshold_var)
            else:
                dim1 = np.sum(var >= threshold_var)

            if dim1 > dim0 and dist[j-1]-dist[j-2] > threshold_edge:
                self.edge_matrix[idx, ind[:j-1]] = dist[:j-1]
                self.edge_matrix[ind[:j-1], idx] = dist[:j-1]

            dim0 = dim1

        self.edge_matrix[idx, ind] = dist
        self.edge_matrix[ind, idx] = dist

    def build_simplex(self, pointcloud, S1=0.2, k=10, threshold_var=0.08, edge_sen=1, **kwargs):
        """
        Computes the 1-skeleton on pointcloud which approximates
        the underlying manifold structure of our data to self.edges
        and self.edge_matrix. Also computes the estimated dimension 
        of our data to self.dim.

        Parameters
        ----------
        pointcloud : (n_samples, n_features) np.array
            The pointcloud data from which we build our simplex.
        k : int
            The number of NN we use.
        threshold_var : [0,1] float
            The threshold to estimate the local intrinsic dimension by PCA.
        edge_sen : positive float
            The sensitivity with which we choose safe edges.
        """

        n = len(pointcloud)
        self.pointcloud = pointcloud
        self.edge_matrix = np.zeros([n, n])

        kd_tree = KDTree(pointcloud, leaf_size=2)
        dists, inds = kd_tree.query(pointcloud, k=k+1)
        # removes points being compared to itself with KNN
        dists = dists[:, 1:]
        inds = inds[:, 1:]

        visible_edges = [self.find_visible_edge(i, inds[i], dists[i]) for i in range(n)]
        dims_vars = [self.find_safe_edges(i, visible_edges[i][0], visible_edges[i][1], threshold_var, edge_sen) for i in range(n)]
        self.edges = [np.where(self.edge_matrix[i]!=0)[0] for i in range(n)]  # ensures can see all edges to i

        local_dims = [local_pca_elbow(pointcloud[edges], S1) for edges in self.edges]  # NOTE : should be translated?
        self.dim = mode(local_dims, axis=None)[0][0]
        self.edge_matrix = csr_matrix(self.edge_matrix)

    def normal_coords(self, k0=0, two_d=False, **kwargs):
        """
        Computes the Riemannian normal coordinates from 
        the 'naive' algorithm, saved to self.coords.

        Parameters
        ----------
        k0 : int
            Maximum number of extra (beyond self.dim) neighbouring 
            points to be used in the projection of each point. 
        two_d : bool
            When True, we set self.dim=2 to compute a 2-D projection.
        """

        if self.edges == None:
            print("No 1-skeleton found! Try build_simplex.")
            return None

        if two_d:
            self.dim = 2

        n = len(self.pointcloud)
        self.coords = np.zeros([n, self.dim])
        computed_points = np.full(n, False)  # tracks which coordinates has been computed

        # find our base point for T_pM
        dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True) 
        p_idx = np.argmin(np.amax(dist_matrix, axis=1))  # assumes connected
        self.p_idx = p_idx
        p = self.pointcloud[p_idx] 
        computed_points[p_idx] = True

        # set up tangent basis
        tangent_inds = np.random.choice(self.edges[p_idx], size=self.dim, replace=False)
        tangent_edges = np.transpose(self.pointcloud[tangent_inds] - p)
        tangent_edges = np.linalg.qr(tangent_edges)[0]  # gives orthonormal basis for T_pM

        # compute normal coords for p's edge points
        edge_points = np.transpose(self.pointcloud[self.edges[p_idx]] - p)
        edge_scalar = np.linalg.norm(edge_points, axis=0)
        edge_coords = np.linalg.lstsq(tangent_edges, edge_points)[0]
        edge_coords = (edge_coords / np.linalg.norm(edge_coords, axis=0)) * edge_scalar
        self.coords[self.edges[p_idx]] = np.transpose(edge_coords)
        computed_points[self.edges[p_idx]] = True

        # then interate over all other points based off of increasing distance from p
        p_dist = dist_matrix[p_idx]
        sorted_inds = np.argsort(p_dist)

        for idx in sorted_inds:
            if computed_points[idx]:
                continue
            else:
                q = self.pointcloud[idx]
                pred = predecessors[p_idx, idx]  # (index of) point before idx on the shortest path from p to idx
                computed_points_b = [i for i in self.edges[pred] if computed_points[i]]

                # we add the indexes of computed points connected to the c_i which are not already in the list and are not b
                if len(computed_points_b) < self.dim+k0:
                    extra_computed_points = [j for i in computed_points_b for j in self.edges[i] if computed_points[j] and j not in computed_points_b and j!= pred]
                    extra_computed_points_idx = np.argsort(dist_matrix[idx, extra_computed_points])
                    computed_points_b += list(np.asarray(extra_computed_points)[extra_computed_points_idx[:k0+self.dim-len(computed_points_b)]])

                k = len(computed_points_b)

                b = self.pointcloud[pred]
                b_prime = self.coords[pred]

                # optimization to solve for the projection of each point
                alpha = np.linalg.norm(q-b)

                y = self.pointcloud[computed_points_b] - b
                y /= np.linalg.norm(y, axis=1).reshape(k, 1) * alpha
                y *= q-b
                y = np.sum(y, axis=1)

                A = self.coords[computed_points_b] - b_prime
                A /= np.linalg.norm(A, axis=1).reshape(k, 1) * alpha  

                m = gp.Model()
                m.setParam('OutputFlag', 0)
                m.setParam(GRB.Param.NonConvex, 2)
                x = m.addMVar(shape=int(self.dim), lb=float('-inf'))
                Q = A.T @ A
                c = -2 * y.T @ A
                obj = x @ Q @ x + c @ x + y.T @ y
                m.setObjective(obj, GRB.MINIMIZE)
                m.addConstr(x@x == alpha**2, name="c")
                m.optimize()
                
                # records the projected coordinates
                self.coords[idx] = x.X + b_prime                    
                computed_points[idx] = True

    def show_boundary(self, alpha, tol, c=None, show_tear_points=False, a=2.5, show_connections=False, show_pointcloud=False, connection_tol=5, **kwargs):
        """
        For 2-D projections, we plot the projection with options with
        options to show the "cleaned" orientation, "tear" points 
        and how boundary points are connected to other boundary points.

        We have the option to plot similar data on the pointcloud (if 3-D).

        Parameters
        ----------
        alpha : float
            Value for alpha shapes.
        tol : int >= 2
            Length of loops to close up with clean_boundary.
        c : colour map
        show_tear_points : bool
            When True, we plot the "tear" points with red crosses.
        a : float
            Value for find_tear_points.
        show_connections : bool
            When True, we show how boundary points are connected
            to other boundary points in the projection.
        show_pointcloud : bool
            When True, we plot the pointcloud with the respective 
            information added.
        connection_tol : non-negative int
            Specifies how far connected points must be to count 
            as a connection.
        """

        if np.all(self.coords == None):
            print('No projection found! Try normal_coords.')
            return None
        elif self.dim != 2:
            print("Dimension of projection is not two - cannot plot boundary!")
            return None

        # computes the boundary and cleans orientation
        boundary_edges = alpha_shape(self.coords, alpha)
        show_orientation = True
        orientation = find_orientation(boundary_edges, **kwargs)
        included = clean_boundary(self, tol, orientation, boundary_edges)

        if np.sum(included) == 2 or np.all(included==-1):  # if all points are identified or orientation is not 1-cycle
            show_orientation = False
            print('Boundary is not a 1-cycle - try a different alpha value!')

        # plot of projections
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(", ".join([i+"="+str(kwargs[i]) for i in kwargs.keys()]) + f',alpha={alpha}, show_tear_points={show_tear_points}, a={a}, dim={self.dim}, n={len(self.coords)}')
        ax = fig.add_subplot(1, 1, 1)
        plt.axis('equal')
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c=c)

        # handles plotting tear points
        if show_tear_points:
            tear_points = find_tear_points(self, a)
            ax.scatter(self.coords[tear_points, 0], self.coords[tear_points, 1], c='r', marker='x', s=50, label=f'boundary with a={a}')
            fig.legend()

        # handles plotting the boundary with orientation
        if show_orientation:
            orientation = orientation[included]
            ax = add_boundary(self, orientation, ax)

        # handles showing the connections of boundary points in the projection
        if show_connections:
            short_connections, _ = intersection(self, orientation[:, 0], connection_tol)
            for idx, short_connection in enumerate(short_connections):
                if len(short_connection) != 0:
                    plt.scatter(self.coords[orientation[:, 0][idx], 0], self.coords[orientation[:, 0][idx], 1], c='r', marker='o')
                    for j in short_connection:
                        plt.plot([self.coords[orientation[:, 0][idx], 0], self.coords[j][0]], [self.coords[orientation[:, 0][idx], 1], self.coords[j][1]], c='black', alpha=0.5)

        # ensures that the pointcloud is 3-D before plotting
        if show_pointcloud and self.pointcloud.shape[-1] != 3:
            print('Pointcloud data is not 3-D - cannot plot!')
            show_pointcloud = False

        # plot of pointcloud in 3-D
        if show_pointcloud:
            fig1 = plt.figure(figsize=(10, 10))
            fig1.suptitle(", ".join([i+"="+str(kwargs[i]) for i in kwargs.keys()]) + f',alpha={alpha}, show_tear_points={show_tear_points}, a={a}, dim={self.dim}, n={len(self.coords)}')
            ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
            ax1.scatter3D(self.pointcloud[:, 0], self.pointcloud[:, 1], self.pointcloud[:, 2], c=c, alpha=0.5)
            ax1.scatter3D(self.pointcloud[self.p_idx, 0], self.pointcloud[self.p_idx, 1], self.pointcloud[self.p_idx, 2], c='g', marker='>', s=100)

            # handles plotting tear points on the pointcloud
            if show_tear_points:
                ax1.scatter3D(self.pointcloud[tear_points, 0], self.pointcloud[tear_points, 1], self.pointcloud[tear_points, 2], c='r', marker='x', s=50, label=f'boundary with a={a}')
                fig1.legend()

            # handles plotting the boundary with orientation on the pointcloud
            if show_orientation:
                boundary_array = np.asarray(list(orientation)).reshape(-1)
                ax1 = add_boundary(self, orientation, ax1, three_d=True)

                # plots the edges for neighbouring points (in the 1-skeleton) of the pointcloud
                for i in range(len(self.pointcloud)):
                    if i in boundary_array:
                        for k in self.edges[i]:
                            if k in boundary_array:
                                ax1.plot3D([self.pointcloud[i][0], self.pointcloud[k][0]],[self.pointcloud[i][1], self.pointcloud[k][1]], [self.pointcloud[i][2], self.pointcloud[k][2]], color='b', alpha=0.3)

        plt.show()

    def compute_quotient_edges(self, alpha, tol, quotient_tol, tol1, connection_tol=5, **kwargs):
        """
        Computes the quotient identifications of the boundary
        of a 2-D projection.

        Parameters
        ----------
        alpha : float
            Value for alpha shapes.
        tol : int >= 2
            Length of loops to close up with clean_boundary.
        quotient_tol : int
            Number of points between short-circuit points
            required (at least) to define a non-short-circuit 
            between them.
        tol1 : int
            Minimum distance required between splitting points 
            when refining edges.
        connection_tol : int
            Specifies how far connected points must be to count 
            as a short-circuit connection.

        Returns
        -------
        short_edges : list of lists
            Each list gives the indexes in clean_orientation of the 
            short-circuit edges.
        non_short_edges : list of lists
            Each list gives the indexes in clean_orientation of the 
            non-connected edges.
        glued_edges : list of lists
            Consist of elements [i, j] where i, j are indexes 
            in short_edges representing "glued" edges.
        orientation : (num_edges, 2) np.array 
            The ith row gives the indexes (j, k) of the connected 
            points of the ith edge in the orientated boundary.
        clean_orientation : (num_edges,) np.array 
            The array lists the indexes of boundary points in order
            (in terms of S.coords).
        same_orientation : bool list
            ith element is True if ith pair in glued_edges have the same
            orientation. False, otherwise.
        orientation_dict : dict
            Keys are indexes for short_edges with value as True if the indexed
            edge should be plotted with the same orientation as induced from 
            clean_orientation. False, otherwise.
        colour_dict : dict
            Keys are indexes for short_edges with int values. Elements with the
            same values are plotted with the same colour - i.e. glued together.
        """

        if np.all(self.coords==None):
            print('No projection found! Try normal_coords.')
            return None
        elif self.dim != 2:
            print("Dimension of projection is not two - cannot compute quotient!")
            return -1

        # set up boundary with orientation
        edges = alpha_shape(self.coords, alpha=alpha)
        orientation = find_orientation(edges)

        # clean up boundary
        included = clean_boundary(self, tol, orientation, edges)
        if np.sum(included) == 2:  # if all points are identified  
            return None, None, None, orientation, orientation[:, 0], None, None, None  # TODO other ways to degenerate
        elif np.all(included==-1):  # if orientation is not 1-cycle
            return -1
        clean_orientation = orientation[:, 0][included]  # 1-D array
        n = len(clean_orientation)

        # set up connective structure of the "cleaned" orientation
        short_connections, short_idxs = intersection(self, clean_orientation, connection_tol)
        m = len(short_idxs)
        if m == 0:  # if only one non-short-circuit 1-cycle 
            return None, [[i for i in range(n)]], None, orientation, clean_orientation, None, None, None

        # identify and refine edges
        short_edges, non_short_edges = identify_edges(n, short_idxs, quotient_tol)
        if len(short_edges) == 0:
            return None, non_short_edges, None, orientation, clean_orientation, None, None, None
        short_edges = refine_edges(short_edges, tol1, clean_orientation, short_connections)

        # glue edges, find their orientation and assemble the quotient
        glued_edges = connect_edges(short_edges, clean_orientation, short_connections)
        same_orientation = gluing_orientation(glued_edges, short_edges, clean_orientation, short_connections)
        if same_orientation == -1:
            return -1
        orientation_dict, colour_dict = assemble_quotient(glued_edges, same_orientation)

        return short_edges, non_short_edges, glued_edges, orientation, clean_orientation, same_orientation, orientation_dict, colour_dict

    def plot_quotient(self, c, alpha, tol, quotient_tol, tol1, connection_tol=5, alpha0=0.8, show_pointcloud=False, **kwargs):
        """
        Plots the quotient identifications of the boundary
        of a 2-D projection, with the option to show the
        correspondence on the pointcloud data (if 3-D).

        Parameters
        ----------
        c : colour map
        alpha : float
            Value for alpha shapes.
        tol : int >= 2
            Length of loops to close up with clean_boundary.
        quotient_tol : int
            Number of points between short-circuit points
            required (at least) to define a non-short-circuit 
            between them.
        tol1 : int
            Minimum distance required between splitting points 
            when refining edges.
        connection_tol : int
            Specifies how far connected points must be to count 
            as a short-circuit connection.
        alpha0 : float
            Alpha value (for line plotting).
        show_pointcloud : bool
            When True, we plot the pointcloud with the respective 
            information added.

        Returns
        -------
        quotient_info : tuple
            Output of compute_quotient_edges.
        """

        if np.all(self.coords == None):
            print('No projection found! Try normal_coords.')
            return None

        # computes the quotient
        quotient_info = self.compute_quotient_edges(alpha, tol, quotient_tol, tol1, connection_tol=connection_tol)
        if quotient_info == -1:
            return None

        # ensures that the pointcloud is 3-D before plotting
        if show_pointcloud and self.pointcloud.shape[-1] != 3:
            print('Pointcloud data is not 3-D - cannot plot!')
            show_pointcloud = False

        if show_pointcloud:
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(1, 2, 1)
            plt.axis('equal')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax1.scatter(self.coords[:, 0], self.coords[:, 1], c=c)
            ax2.scatter3D(self.pointcloud[:, 0], self.pointcloud[:, 1], self.pointcloud[:, 2], c=c)
            ax2.scatter3D(self.pointcloud[self.p_idx, 0], self.pointcloud[self.p_idx, 1], self.pointcloud[self.p_idx, 2], c='g', marker='>', s=100)
        else:
            fig = plt.figure(figsize=(10, 10))
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.scatter(self.coords[:, 0], self.coords[:, 1], c=c)
            plt.axis('equal')

        # setting up fig and plotting colours
        fig.suptitle(", ".join([i+"="+str(kwargs[i]) for i in kwargs.keys()]) + f'alpha={alpha}, tol={tol}, quotient_tol={quotient_tol}, tol1={tol1},  n={len(self.coords)}')
        orientation_cmaps = ['cool', 'Purples', 'Blues', 'Greens', 'Oranges',
                        'YlOrBr', 'YlOrRd', 'OrRd', 'Greys', 'PuRd', 'RdPu', 
                        'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        colours = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'cyan']

        short_edges, non_short_edges, glued_edges, orientation, clean_orientation, same_orientation, orientation_dict, colour_dict = quotient_info
        clean_orientation_2d = convert_orientation(clean_orientation)

        # handles looping of boundary orientation if no short-edges exist 
        no_short_edges = False
        if short_edges == None:
            no_short_edges = True
        else:
            if len(short_edges) == 0:
                no_short_edges = True

        # handles looping of boundary orientation if no non-short-edges exist 
        no_non_short_edges = False
        if non_short_edges == None:
            no_non_short_edges = True
        else:
            if len(non_short_edges) == 0:
                no_non_short_edges = True

        # handles non-short-edges
        if non_short_edges != None:
            if len(non_short_edges) == 1 and no_short_edges:  # if single non-short-circuit 1-cycle 
                ax1 = add_boundary(self, clean_orientation_2d, ax1, three_d=False, alpha0=alpha0)
                if show_pointcloud:
                    ax2 = add_boundary(self, clean_orientation_2d, ax2, three_d=True, alpha0=alpha0)
            else:
                for idx, non_short_edge in enumerate(non_short_edges):
                    edge = clean_orientation_2d[non_short_edge]
                    l = len(edge)
                    for i, j in edge:
                        ax1.plot([self.coords[i][0], self.coords[j][0]],[self.coords[i][1], self.coords[j][1]], color=colours[idx], linewidth=10, alpha=alpha0, linestyle='dotted')
                        if show_pointcloud:
                            ax2.plot3D([self.pointcloud[i][0], self.pointcloud[j][0]],[self.pointcloud[i][1], self.pointcloud[j][1]], [self.pointcloud[i][2], self.pointcloud[j][2]], color=colours[idx], linewidth=10, alpha=alpha0, linestyle='dotted')

        # handles short-edges
        if short_edges != None:
            if len(short_edges) == 1 and no_non_short_edges == True:
                loop = True
            else:
                loop = False
            for idx, short_edge in enumerate(short_edges):
                if orientation_dict[idx]:
                    short_edge_clean_orientation = clean_orientation_2d[short_edge]
                else:
                    short_edge_clean_orientation = np.flip(clean_orientation_2d[short_edge])

                ax1 = add_boundary(self, short_edge_clean_orientation, ax1, three_d=False, alpha0=alpha0, cmap=orientation_cmaps[colour_dict[idx]], loop=loop)
                if show_pointcloud:
                    ax2 = add_boundary(self, short_edge_clean_orientation, ax2, three_d=True, alpha0=alpha0, cmap=orientation_cmaps[colour_dict[idx]], loop=loop)

         # handles if all points are associated 
        if non_short_edges == None and short_edges == None:
            for i, j in clean_orientation_2d:
                ax1.plot([self.coords[i][0], self.coords[j][0]],[self.coords[i][1], self.coords[j][1]], color='black', linewidth=10, alpha=alpha0)
                if show_pointcloud:
                    ax2.plot3D([self.pointcloud[i][0], self.pointcloud[j][0]],[self.pointcloud[i][1], self.pointcloud[j][1]], [self.pointcloud[i][2], self.pointcloud[j][2]], color='black', linewidth=10, alpha=alpha0)

        plt.show()

        return quotient_info


if __name__ == '__main__':
    pass