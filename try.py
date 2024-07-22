import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Load data and model
PATH = 'data_from_inbal/100X100/'

N = 100

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
model = load_model("trained_model/100*100/weights_2d.keras")


def plot_row(example_num, row, actual_output, predicted_output=None):
    """
    Plots a 30x30 grid map and the actual path with heatmaps for actual and predicted outputs.

    Parameters:
    - row: A 30x30x3 numpy array representing the grid map.
    - actual_output: A 30x30 numpy array representing the actual path.
    - predicted_output: A 30x30 numpy array representing the predicted path (optional).
    """

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
        predicted_output = np.where(predicted_output > 0.3, predicted_output, np.nan)
        sns.heatmap(predicted_output, annot=True, cmap='YlGnBu', ax=axs[1], annot_kws={"size": 8})

        # Add obstacles to the heatmap
        for i in range(N):
            for j in range(N):
                if row[:, :, 0][i, j] == 1:
                    axs[1].add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

        axs[1].set_title('Predicted Data'+str(example_num))
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')

    plt.tight_layout()
    plt.show()


for _ in range(4418, 4430):
    row_number = int(_)

    prediction = model.predict(x3d[row_number].reshape(1, N, N, 3))
    plot_row(row_number, x3d[row_number], y[row_number], prediction[0, :, :, 0])

tf.keras.backend.clear_session()
