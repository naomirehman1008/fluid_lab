import networkx as nx
import numpy as np
from toponetx import SimplicialComplex
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def grid_to_graph(grid):
    '''
    Input: a MxNxd NumPy array representing a fluid field where each coordinate has d attributes (velocity, pressure, etc.)
    Returns: 
    - A NetworkX graph with MxN nodes. Edges triangulate the grid; each node at index (i,j) has an edge between itself 
        and the node at (i+1, j), (i, j+1), (i+1, j+1), but not (i+1, j-1)
    - coords_to_node: a function that takes in an (i,j) pair and returns the node ID in the graph
    - node_to_coords: a function that takes in a node ID in the graph and returns its (i,j) coordinates
    '''
    M, N, _ = grid.shape
    G = nx.Graph()

    coords_to_node = lambda i, j: i * N + j
    node_to_coords = lambda node: (node // N, node % N)

    for i in range(M):
        for j in range(N):
            node = coords_to_node(i, j)
            G.add_node(node, attr=grid[i, j])

    for i in range(M):
        for j in range(N):
            node = coords_to_node(i, j)
            if i + 1 < M:
                G.add_edge(node, coords_to_node(i + 1, j))       # vertical
            if j + 1 < N:
                G.add_edge(node, coords_to_node(i, j + 1))       # horizontal
            if i + 1 < M and j + 1 < N:
                G.add_edge(node, coords_to_node(i + 1, j + 1))   # diagonal down-right

    return G, coords_to_node, node_to_coords
    

def _graph_to_complex(G, M=None, N=None):
    """
    Construct the simplicial complex directly from a grid graph G
    built by grid_to_graph, without calling graph_to_clique_complex.
    """
    if M is None or N is None:
        # infer grid shape from number of nodes (works since node ids are 0..M*N-1)
        num_nodes = G.number_of_nodes()
        M = int(np.sqrt(num_nodes))
        N = num_nodes // M

    SC = SimplicialComplex()

    # Add 0-simplices (nodes)
    for v in G.nodes:
        SC.add_simplex([v])

    # Add 1-simplices (edges)
    for u, v in G.edges:
        SC.add_simplex([u, v])

    # Add 2-simplices (triangles)
    # We know each square forms one triangle (i,j)-(i+1,j)-(i,j+1)
    # and one diagonal (i+1,j+1) completes the second triangle.
    def coords_to_node(i, j):
        return i * N + j

    for i in range(M - 1):
        for j in range(N - 1):
            v00 = coords_to_node(i, j)
            v10 = coords_to_node(i + 1, j)
            v01 = coords_to_node(i, j + 1)
            v11 = coords_to_node(i + 1, j + 1)

            # Two triangles per cell (consistent with your edge pattern)
            SC.add_simplex([v00, v10, v11])
            SC.add_simplex([v00, v01, v11])

    return SC

class GridRepresentations:
    '''
    This class handles several representations of a fluid flow.
    It takes a grid as input, and constructs graph and simplicial complex representations of that flow grid.
    It also constructs node, edge, and 2-simplex features matrices.
    All matrices and combinatorial objects are kept on the level of NumPy/NetworkX/TopoNetX.
    '''
    def __init__(self, grid):
        self.velo_grid = grid #nxmxd grid of velocities at each point
        self.G, self.coords_to_node, self.node_to_coords = grid_to_graph(self.velo_grid) # NetworkX graph representation
        # And also functions mapping coordinates to node IDs and back
        # self.K = _graph_to_complex(self.G) # TopoNetX simplicial complex representation
        # self.x_0 = self._positions() # node-level features (np.array)
        # self.x_1 = self.edge_velocities() # edge-level features (np.array)
        # self.x_2 = self.face_circulations() # face-level features (np.array)

def positions(grid):
    '''
    Given an MxNxd NumPy array, returns an MxNx2 matrix giving the (i,j)
    coordinates of each grid point. The last axis of the input (velocity etc.)
    is ignored.
    '''
    M, N, _ = grid.velo_grid.shape
    i_coords, j_coords = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    positions = np.stack((i_coords, j_coords), axis=-1)  # shape (M, N, 2)
    return positions
    
    def node_energy_features(self):
        '''
        Returns a list of node-level features, where each feature is the squared
        total velocity magnitude at that node (i.e., ||v||²).
        '''
        energies = []
        for node in self.G.nodes():
            i, j = self.node_to_coords(node)
            v = self.velo_grid[i, j]
            energies.append(np.dot(v, v))  # squared velocity magnitude
        return energies

def edge_velocities(G, grid, node_to_coords, coord_to_nodes):
    '''
    Computes edge-level features according to the indexing imposed by self.K,
    using the function _edge_velocity.
    Returns an array of shape (num_edges, 1) with each edge’s velocity feature.
    '''
    edge_features = []
    for u, v in tqdm(grid.edges()):
        edge_features.append(_edge_velocity(u, v))
    return np.array(edge_features)

def _edge_velocity(G, grid, node_to_coords, coord_to_nodes, u, v):
    '''
    Given nodes u,v, compute an edge-level velocity feature by projecting
    the node velocities onto the unit vector along the edge, then averaging.

    Should check that (u,v) is a 1-simplex in self.K and raise an assertion if not.
    '''
    assert (u, v) in G.edges() or (v, u) in G.edges(), \
        f"({u},{v}) is not a valid edge in the graph."

    # Coordinates of the nodes
    i1, j1 = node_to_coords(u)
    i2, j2 = node_to_coords(v)

    # Direction vector (from u to v)
    direction = np.array([i2 - i1, j2 - j1], dtype=float)
    direction /= np.linalg.norm(direction)  # unit vector

    # Velocities at nodes
    vel_u = grid[i1, j1][:2]
    vel_v = grid[i2, j2][:2]

    # Project both velocities onto direction and average
    proj_u = np.dot(vel_u, direction)
    proj_v = np.dot(vel_v, direction)
    return 0.5 * (proj_u + proj_v)

def face_circulations(K):
    '''
    Computes 2-simplex-level features (face circulations) according to
    the indexing in self.K, using _face_circulation.
    Returns an array of shape (num_faces, 1).
    '''
    face_features = []
    for simplex in tqdm([simplex for simplex in K.simplices if len(simplex)==3]):  # 2-simplices (triangles)
        x1, x2, x3 = simplex
        face_features.append(_face_circulation(x1, x2, x3))
    return np.array(face_features)

def _face_circulation(x_1, x_2, x_3):
    '''
    Given nodes x_1, x_2, x_3 forming a 2-simplex (triangle),
    computes a scalar circulation feature around the face.
    Should check that (x_1, x_2, x_3) is indeed a 2-simplex in self.K.
    '''
    # simplex = {x_1, x_2, x_3}
    # assert simplex in [set(simplex) for simplex in self.K.simplices if len(simplex)==3], \
    #     f"({x_1},{x_2},{x_3}) is not a 2-simplex in the complex."

    # Get edge circulation around triangle (sum of edge projections)
    edges = [(x_1, x_2), (x_2, x_3), (x_3, x_1)]
    circ = 0.0
    for u, v in edges:
        circ += _edge_velocity(u, v)
    return circ
    
def visualize_energy(graph, grid, node_to_coords, ax=None, node_size=30, edge_width=1.5):
    pos = {n: node_to_coords(n)[::-1] for n in graph.nodes()}  # (x, y) layout

    n_nodes = len(graph.nodes())
    fig_size = max(4, np.sqrt(n_nodes) / 6)

    # Create axis if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        created_fig = True

    nx.draw(
        graph, pos,
        node_size=node_size,
        node_color=grid.flatten(),
        edge_color="#444444",
        width=edge_width,
        cmap='viridis',
        ax=ax
    )
    ax.set_title("Graph colored by squared total velocity")
    ax.set_aspect("equal")

    # Only apply tight layout and show if we created the figure
    if created_fig:
        plt.tight_layout()
        plt.show()