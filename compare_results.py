import tensorflow as tf
from keras.models import load_model
import os
# Load data and model
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'data_from_inbal/30X30/'

N = 30

def load_data(input_path: str):
    x = np.loadtxt(os.path.join(input_path, 'inputs.dat'))
    s_map = np.loadtxt(os.path.join(input_path, 's_maps.dat'))
    g_map = np.loadtxt(os.path.join(input_path, 'g_maps.dat'))
    y = np.loadtxt(os.path.join(input_path, 'outputs.dat'))
    return x, s_map, g_map, y

def preprocess_data(x, s_map, g_map, y):
    m = x.shape[0]
    n = int(np.sqrt(x.shape[1]))
    x = x.reshape(m, n, n)
    s_map = s_map.reshape(m, n, n)
    g_map = g_map.reshape(m, n, n)
    y = y.reshape(m, n, n)
    x3d = np.stack((x, s_map, g_map), axis=-1)
    return x3d, y

x, s_map, g_map, y = load_data(PATH)
x3d, y = preprocess_data(x, s_map, g_map, y)
model = load_model("weights_2d.keras")



def is_path_present(start, goal, thresholded_pred):
    """Check if there's a path between start and goal using BFS with diagonal movements."""
    start_coords = np.argwhere(start == 1)[0]
    goal_coords = np.argwhere(goal == 1)[0]

    queue = deque([(tuple(start_coords), 0)])  # The queue stores tuples of coordinates and path length
    visited = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Include diagonals

    while queue:
        (y, x), path_length = queue.popleft()

        if (y, x) == tuple(goal_coords):
            return path_length + 1  # Path exists and return the path length

        if (y, x) in visited:
            continue

        visited.add((y, x))

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < thresholded_pred.shape[0] and 0 <= nx < thresholded_pred.shape[1] and \
                    thresholded_pred[ny, nx] > THRESHOLD and (ny, nx) not in visited:
                queue.append(((ny, nx), path_length + 1))

    return -1  # No path exists


def plot_row_th(example_num, row, actual_output, predicted_output=None):
    """
    Plots a 30x30 grid map and the actual path with heatmaps for actual and predicted outputs.

    Parameters:
    - row: A 30x30x3 numpy array representing the grid map.
    - actual_output: A 30x30 numpy array representing the actual path.
    - predicted_output: A 30x30 numpy array representing the predicted path (optional).
    """
    if PLOT:
        # Set up the matplotlib figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Function to plot the grid map
        def plot_grid(ax, matrix, title, path=None):
            for i in range(N):
                for j in range(N):
                    if matrix[:, :, 0][i, j] == 1:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))  # Obstacle
                    elif matrix[:, :, 1][i, j] == 1:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))  # Start
                    elif matrix[:, :, 2][i, j] == 1:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red'))  # Goal
                    if path is not None and path[i, j] == 1:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='blue'))  # Path

            ax.set_aspect('equal')
            ax.set_xlim(0, N)
            ax.set_ylim(0, N)
            ax.invert_yaxis()
            ax.set_title(title)

        # Plot the grid map with the actual path
        plot_grid(axs[0], row, 'Actual Path', actual_output)

        # Create the heatmap for predicted data if provided
        if predicted_output is not None:
            predicted_output = np.where(predicted_output > THRESHOLD, predicted_output, np.nan)
            sns.heatmap(predicted_output, annot=True, cmap='YlGnBu', ax=axs[1], annot_kws={"size": 8})

            # Add obstacles to the heatmap
            for i in range(N):
                for j in range(N):
                    if row[:, :, 0][i, j] == 1:
                        axs[1].add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

            axs[1].set_title('Predicted Data' + str(example_num))
            axs[1].set_xlabel('X-axis')
            axs[1].set_ylabel('Y-axis')

            plt.tight_layout()
            plt.show()
    if predicted_output is not None:
        path_exists = is_path_present(row[:, :, 1], row[:, :, 2], predicted_output)
        actual_length = int(np.sum(actual_output))
        print("Predicted path length:", path_exists, "Actual path length:", actual_length)

    return path_exists, actual_length




# Function to count paths found and not found
def count_paths(df):
    count_not_found = (df['Predicted_Path'] == -1).sum()
    count_found = (df['Predicted_Path'] > -1).sum()
    return count_not_found, count_found


# Function to create and plot path status counts with percentages
def plot_path_status(count_not_found, count_found, N_tries):
    count_data = pd.DataFrame({
        'Path_Status': ['Not Found', 'Found'],
        'Count': [count_not_found, count_found]
    })
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='Path_Status', y='Count', data=count_data)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / N_tries)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.title('Number of Paths Found vs. Not Found')
    plt.xlabel('Path Status')
    plt.ylabel('Count')
    plt.show()


# Function to compare predicted paths with actual paths
def compare_paths(found_paths):
    count_less = (found_paths['Predicted_Path'] < found_paths['Actual_Path']).sum()
    count_equal = (found_paths['Predicted_Path'] == found_paths['Actual_Path']).sum()
    count_longer = (found_paths['Predicted_Path'] > found_paths['Actual_Path']).sum()
    return count_less, count_equal, count_longer


# Function to create and plot comparison counts with percentages
def plot_comparison(count_less, count_equal, count_longer, count_found):
    comparison_data = pd.DataFrame({
        'Comparison': ['Less than Actual', 'Equal to Actual', 'Longer than Actual'],
        'Count': [count_less, count_equal, count_longer]
    })
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='Comparison', y='Count', data=comparison_data)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / count_found)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.title('Comparison of Predicted Path Lengths to Actual Paths')
    plt.xlabel('Comparison')
    plt.ylabel('Count')
    plt.show()


# Function to print summary statistics
def print_summary_statistics(count_not_found, count_found, count_equal, count_longer, N_tries):
    paths_found = count_found / N_tries
    paths_not_found = count_not_found / N_tries
    paths_are_equal = count_equal / count_found
    paths_are_longer = count_longer / count_found

    print("How many paths_found:", paths_found)
    print("How many paths_not_found:", paths_not_found)
    print("How many paths_are_equal:", paths_are_equal)
    print("How many paths_are _longer:", paths_are_longer)

    return paths_found, paths_not_found, paths_are_equal, paths_are_longer


# Main function to execute the analysis
def main(result, N_tries):
    # Create a DataFrame to store the results
    df = pd.DataFrame(result, columns=['Predicted_Path', 'Actual_Path'])

    # Count paths found and not found
    count_not_found, count_found = count_paths(df)

    # Filter the rows where the path is found
    found_paths = df[df['Predicted_Path'] > -1]

    # Compare predicted paths with actual paths
    count_less, count_equal, count_longer = compare_paths(found_paths)

    if PLOT:
        # Plot path status
        plot_path_status(count_not_found, count_found, N_tries)

        # Plot comparison of paths
        plot_comparison(count_less, count_equal, count_longer, count_found)

    # Print summary statistics
    return print_summary_statistics(count_not_found, count_found, count_equal, count_longer, N_tries)


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_tries = 2000
PLOT=False
result = np.zeros([N_tries,2])
threshold_results = np.zeros([len(THRESHOLDS), 4])
for idx, t in enumerate(THRESHOLDS):
    THRESHOLD = t
    for i in range(1, N_tries):
        prediction = model.predict(x3d[i].reshape(1, N, N, 3))
        result[i] = plot_row_th(example_num=i, row=x3d[i], actual_output=y[i], predicted_output=prediction[0, :, :, 0]) #  return path_exists, actual_length
    threshold_results[idx] = main(result, N_tries)

threshold_results_df = pd.DataFrame(threshold_results, columns=["paths_found", "paths_not_found", "paths_are_equal", "paths_are_longer"])
threshold_results_df.index = THRESHOLDS

# Plotting paths_found and paths_not_found
plt.figure(figsize=(12, 6))

# Plot paths_found
plt.plot(threshold_results_df.index, threshold_results_df['paths_found'], marker='o', linestyle='-', color='b', label='Paths Found')

# Plot paths_not_found
plt.plot(threshold_results_df.index, threshold_results_df['paths_not_found'], marker='o', linestyle='-', color='g', label='Paths Not Found')

# Add labels and title
plt.xlabel('Threshold')
plt.ylabel('Count - %')
plt.title('Paths Found vs. Paths Not Found')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Plotting paths_are_equal and paths_are_longer
plt.figure(figsize=(12, 6))

# Plot paths_are_equal
plt.plot(threshold_results_df.index, threshold_results_df['paths_are_equal'], marker='o', linestyle='-', color='r', label='Paths Are Equal')

# Plot paths_are_longer
#plt.plot(threshold_results_df.index, threshold_results_df['paths_are_longer'], marker='o', linestyle='-', color='purple', label='Paths Are Longer')

# Add labels and title
plt.xlabel('Threshold')
plt.ylabel('Count - %')
plt.title('How Many Paths Are Equal to Actual Paths')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

tf.keras.backend.clear_session()