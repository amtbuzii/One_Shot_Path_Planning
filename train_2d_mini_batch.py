from typing import Tuple, List
import keras
import os
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up TensorFlow GPU configuration at the start
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)
else:
    print("No GPUs available")


EPOCHS = 50
PATH = 'data_from_inbal/100X100/'
HIDDEN_LAYERS = 5
TRAIN_RATIO = 0.7
N = 100

directories_path = [f'database/generated_environments_chunk_{i}' for i in range(1, 5)]

def load_data_batch(directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a batch of data from the given directory.

    Args:
    directory: Path to the directory containing data files.

    Returns:
    Four NumPy arrays containing input data, start maps, goal maps, and outputs.
    """
    x = np.loadtxt(os.path.join(directory, 'inputs.dat'))
    s_map = np.loadtxt(os.path.join(directory, 's_maps.dat'))
    g_map = np.loadtxt(os.path.join(directory, 'g_maps.dat'))
    y = np.loadtxt(os.path.join(directory, 'outputs.dat'))

    return x, s_map, g_map, y


def preprocess_data(x: np.ndarray, s_map: np.ndarray, g_map: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess input data, start maps, goal maps, and outputs.

    Args:
    x: Input data.
    s_map: Start maps.
    g_map: Goal maps.
    y: Outputs.

    Returns:
    Preprocessed data in the format (x_train, y_train, x_test, y_test).
    """
    m = x.shape[0]  # Number of samples
    n = int(np.sqrt(x.shape[1]))  # Dimension of each sample

    x = x.reshape(m, n, n)
    s_map = s_map.reshape(m, n, n)
    g_map = g_map.reshape(m, n, n)
    y = y.reshape(m, n, n)

    x3d = np.stack((x, s_map, g_map), axis=-1)

    return x3d, y


def build_model(input_shape: Tuple[int, int, int]) -> Model:
    """
    Build and compile the neural network model.

    Args:
    input_shape: Shape of the input data.

    Returns:
    Compiled neural network model.
    """
    x = Input(shape=input_shape)

    net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(x)
    net = BatchNormalization()(net)
    for _ in range(HIDDEN_LAYERS):
        net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
        net = BatchNormalization()(net)

    net = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='sigmoid')(net)
    net = BatchNormalization()(net)
    net = Dropout(0.10)(net)

    model = Model(inputs=x, outputs=net)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


def train_model_on_batch(model: Model, x_batch: np.ndarray, y_batch: np.ndarray) -> keras.callbacks.History:
    """
    Train the neural network model on a batch of data.

    Args:
    model: Compiled neural network model.
    x_batch: Batch input data.
    y_batch: Batch output data.

    Returns:
    History object containing training metrics.
    """
    history = model.fit(x_batch, y_batch, batch_size=32, epochs=EPOCHS, verbose=1)
    return history


def save_model(model: Model, filename: str) -> None:
    """
    Save the trained neural network model to a file.

    Args:
    model: Trained neural network model.
    filename: Name of the file to save the model.
    """
    model.save(filename)


def split_data(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the input data and labels into training and testing sets.

    Args:
    x: Input data.
    y: Output data.
    train_ratio: Ratio of data to be allocated for training. Defaults to 0.7.

    Returns:
    Four NumPy arrays containing training input, training output, testing input, and testing output.
    """
    assert 0 < train_ratio < 1, "Train ratio must be between 0 and 1"

    split_index = int(len(x) * train_ratio)
    print("split index", split_index)

    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]

    return x_train, y_train, x_test, y_test


# Define colors
colors = ['black', 'green', 'red', 'blue']  # obstacle, start, goal, path


def plot_row(row, actual_output, predicted_output=None):
    """
    Plots a 30x30 grid map and the actual path with heatmaps for actual and predicted outputs.

    Parameters:
    - row: A 30x30x3 numpy array representing the grid map.
    - actual_output: A 30x30 numpy array representing the actual path.
    - predicted_output: A 30x30 numpy array representing the predicted path (optional).
    """

    # Set up the matplotlib figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

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

    # Plot the grid map without the path
    plot_grid(axs[0], row, 'Map')

    # Plot the grid map with the actual path
    plot_grid(axs[1], row, 'Actual Path', actual_output)

    plt.show()

    # Create heatmaps for actual and predicted outputs if predicted_output is provided
    if predicted_output is not None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Create the heatmap for actual data
        sns.heatmap(actual_output, annot=True, cmap='YlGnBu', ax=axs[0])
        axs[0].set_title('Actual Data')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')

        # Create the heatmap for predicted data
        sns.heatmap(predicted_output, annot=True, cmap='YlGnBu', ax=axs[1])
        axs[1].set_title('Predicted Data')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print('Build model ...')
    model = build_model(input_shape=(N, N, 3))
    model.summary()

    print('Train network ...')
    for directory in directories_path:
        print(f'Loading data from {directory} ...')
        x, s_map, g_map, y = load_data_batch(directory)
        x3d, y = preprocess_data(x, s_map, g_map, y)

        print('Train model on current batch ...')
        train_model_on_batch(model, x3d, y)

        print('Save model after current batch ...')
        save_model(model, "trained_model/100*100/model_2d.keras")

    print('Test network ...')
    x_test, s_map_test, g_map_test, y_test = load_data_batch(directories_path[-1])
    x3d_test, y_test = preprocess_data(x_test, s_map_test, g_map_test, y_test)
    model = load_model("trained_model/100*100/model_2d.keras")
    loss, accuracy = model.evaluate(x3d_test, y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    tf.keras.backend.clear_session()
