import numpy as np
import random
import networkx as nx
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def create_grid(size, num_obstacles, obstacle_size):
    """Create a grid with large, contiguous obstacles."""
    grid = np.zeros((size, size), dtype=int)
    for _ in range(num_obstacles):
        x_start = random.randint(0, size - obstacle_size)
        y_start = random.randint(0, size - obstacle_size)
        grid[x_start:x_start + obstacle_size, y_start:y_start + obstacle_size] = 1
    return grid


def place_start_goal(grid, num_pairs, min_distance):
    """Place multiple start and goal points in the grid with constraints."""
    size = grid.shape[0]
    free_cells = [(i, j) for i in range(size) for j in range(size) if grid[i, j] == 0]

    if len(free_cells) < 2 * num_pairs:
        raise ValueError("Not enough free cells to place start and goal points.")

    def distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    start_points = []
    goal_points = []

    while len(start_points) < num_pairs:
        start = random.choice(free_cells)
        free_cells.remove(start)
        valid_goal = False

        while not valid_goal and free_cells:
            goal = random.choice(free_cells)
            free_cells.remove(goal)
            if distance(start, goal) >= min_distance:
                start_points.append(start)
                goal_points.append(goal)
                valid_goal = True

    return start_points, goal_points


def add_edges_with_weights(G, size):
    """Add edges to the graph with weights considering diagonal moves."""
    for i in range(size):
        for j in range(size):
            if (i, j) not in G.nodes:
                continue
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size and (ni, nj) in G.nodes:
                        weight = 1 if (di == 0 or dj == 0) else 1.414
                        G.add_edge((i, j), (ni, nj), weight=weight)


def a_star_networkx(grid, start, goal):
    """A* algorithm using NetworkX to find the shortest path from start to goal."""
    size = grid.shape[0]
    G = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
    for (i, j), value in np.ndenumerate(grid):
        if value == 1:
            G.remove_node((i, j))
    add_edges_with_weights(G, size)

    try:
        path = nx.astar_path(G, start, goal, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None


def generate_single_environment(args):
    num_envs, grid_size, num_obstacles, obstacle_size, num_start_goal_pairs, min_distance, i = args
    set_random_seed(i)  # Set the random seed based on the environment index for reproducibility
    grid = create_grid(grid_size, num_obstacles, obstacle_size)
    start_points, goal_points = place_start_goal(grid, num_start_goal_pairs, min_distance)

    paths = []
    for start, goal in zip(start_points, goal_points):
        path = a_star_networkx(grid, start, goal)
        paths.append(path)

    s_maps = np.zeros(grid_size * grid_size, dtype=int)
    for start in start_points:
        index = start[0] * grid_size + start[1]
        s_maps[index] = 1

    g_maps = np.zeros(grid_size * grid_size, dtype=int)
    for goal in goal_points:
        index = goal[0] * grid_size + goal[1]
        g_maps[index] = 1

    inputs = grid.flatten()

    output = np.zeros(grid_size * grid_size, dtype=int)
    for path in paths:
        if path:
            for (x, y) in path:
                index = x * grid_size + y
                output[index] = 1

    return s_maps, g_maps, inputs, output


def write_chunk_to_files(chunk_data, chunk_index, grid_size):
    """Write a chunk of environment data to .dat files."""
    s_maps_list, g_maps_list, inputs_list, output_list = chunk_data

    def write_dat_file(filename, data):
        with open(filename, 'w') as file:
            for row in data:
                file.write(' '.join(map(str, row)) + '\n')

    flat_size = grid_size * grid_size

    s_maps_array = np.array(s_maps_list)
    g_maps_array = np.array(g_maps_list)
    inputs_array = np.array(inputs_list)
    output_array = np.array(output_list)

    s_maps_flat = s_maps_array.reshape(-1, flat_size)
    g_maps_flat = g_maps_array.reshape(-1, flat_size)
    inputs_flat = inputs_array.reshape(-1, flat_size)
    output_flat = output_array.reshape(-1, flat_size)

    output_dir = f"generated_environments_chunk_{30+chunk_index}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_dat_file(os.path.join(output_dir, "s_maps.dat"), s_maps_flat)
    write_dat_file(os.path.join(output_dir, "g_maps.dat"), g_maps_flat)
    write_dat_file(os.path.join(output_dir, "inputs.dat"), inputs_flat)
    write_dat_file(os.path.join(output_dir, "outputs.dat"), output_flat)


def generate_environments(num_envs, grid_size, num_obstacles, obstacle_size, num_start_goal_pairs, min_distance,
                          chunk_size):
    print("Starting environment generation...")

    num_chunks = num_envs // chunk_size
    if num_envs % chunk_size > 0:
        num_chunks += 1

    for chunk_index in range(num_chunks):
        print(f"Generating chunk {chunk_index + 1}/{num_chunks}...")

        s_maps_list = []
        g_maps_list = []
        inputs_list = []
        output_list = []

        start_index = chunk_index * chunk_size
        end_index = min(start_index + chunk_size, num_envs)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(generate_single_environment, (
            num_envs, grid_size, num_obstacles, obstacle_size, num_start_goal_pairs, min_distance, i)) for i in
                       range(start_index, end_index)]

            for i, future in enumerate(as_completed(futures), start_index):
                try:
                    s_maps, g_maps, inputs, output = future.result()
                    s_maps_list.append(s_maps)
                    g_maps_list.append(g_maps)
                    inputs_list.append(inputs)
                    output_list.append(output)

                    if (i + 1) % 1000 == 0:
                        print(
                            f"Processed {i + 1 - start_index}/{chunk_size} environments in chunk {chunk_index + 1}...")

                except Exception as e:
                    print(f"Error processing environment {i}: {e}")

        write_chunk_to_files((s_maps_list, g_maps_list, inputs_list, output_list), chunk_index + 1, grid_size)

    print("Environment generation complete.")


def main():
    num_envs = 100  # Total number of environments to generate
    grid_size = 100
    num_obstacles = 12
    obstacle_size = 25
    num_start_goal_pairs = 1
    min_distance = 20
    chunk_size = 50  # Number of environments per file

    set_random_seed(42)  # Set a fixed seed for reproducibility
    generate_environments(num_envs, grid_size, num_obstacles, obstacle_size, num_start_goal_pairs, min_distance,
                          chunk_size)


if __name__ == "__main__":
    main()
