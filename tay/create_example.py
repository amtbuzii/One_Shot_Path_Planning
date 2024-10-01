import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Constants
N = 100  # Adjust as needed


def load_data():
    goal_position = np.loadtxt('/home/amitbou/One_Shot_Path_Planning/database/5K/generated_environments_chunk_1/g_maps.dat')
    obstacle_map = np.loadtxt('/home/amitbou/One_Shot_Path_Planning/database/5K/generated_environments_chunk_1/inputs.dat')
    return obstacle_map, goal_position


def preprocess_data(obstacle_map, goal_position):
    m = obstacle_map.shape[0]
    n = int(np.sqrt(obstacle_map.shape[1]))
    obstacle_map = obstacle_map.reshape(m, n, n)
    goal_position = goal_position.reshape(m, n, n)
    return obstacle_map, goal_position


def create_obs_scenario(ind):  # Run this in order to add the gradients to 'ind' scenario
    obs = x[ind, :, :]
    to = g_maps[ind, :, :]

    flat_max_idx = np.argmax(to)
    row_idx = flat_max_idx // to.shape[1]  # Integer division to get row index
    col_idx = flat_max_idx % to.shape[1]  # Modulo to get column index within the row
    to_tuple = (row_idx, col_idx)

    dst, vst = dijkstra_distances(to_tuple, obs)
    dst[dst == -1] = np.max(dst) + 10  # For not going through obstacles

    return obs, to, dst, vst


def grid_to_graph(obs):
    """Convert grid into a graph suitable for Dijkstra."""
    graph = np.inf * np.ones((N * N, N * N))  # Create a full graph with infinite distances

    for row in range(N):
        for col in range(N):
            if obs[row, col] == 0:  # Only consider free cells
                node = row * N + col
                # Check neighbors (up, down, left, right)
                for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nrow, ncol = row + drow, col + dcol
                    if 0 <= nrow < N and 0 <= ncol < N and obs[nrow, ncol] == 0:
                        neighbor_node = nrow * N + ncol
                        graph[node, neighbor_node] = 1  # Distance of 1 between neighbors

    return csr_matrix(graph)


def dijkstra_distances(goal, obs):
    """Compute shortest paths using Dijkstra from the goal position."""
    graph = grid_to_graph(obs)
    goal_node = goal[0] * N + goal[1]  # Convert goal to a graph node index

    distances, predecessors = dijkstra(csgraph=graph, directed=False, indices=goal_node, return_predecessors=True)

    # Reshape distances to grid format
    dst = distances.reshape(N, N)

    # Create the visited array from the predecessors
    vst = np.zeros_like(dst)
    vst[dst != np.inf] = 1  # Mark nodes with finite distances as visited

    return dst, vst


obstacle_map, goal_position = load_data()
x, g_maps = preprocess_data(obstacle_map, goal_position)
del obstacle_map, goal_position

db_size = x.shape[0]
inp = np.zeros((db_size, N, N, 2))
target = np.zeros((db_size, N, N))

for ind in range(50):
    obs, to, dst, vst = create_obs_scenario(ind)
    inp[ind, :, :, 0] = obs
    inp[ind, :, :, 1] = to
    target[ind, :, :] = dst
    print(ind / db_size)

np.savez_compressed('amit_compressed.npz', inp=inp, target=target)
