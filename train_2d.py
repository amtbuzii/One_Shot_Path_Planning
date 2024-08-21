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
HIDDEN_LAYERS = 100
TRAIN_RATIO = 0.7
N = 100
directories_path = [f'database/generated_environments_chunk_{i}' for i in range(1, 41)]

def load_data_from_directories(directories: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load input data, start maps, goal maps, and outputs from multiple directories.

    Args:
    directories: List of paths to directories containing data files.

    Returns:
    Four concatenated NumPy arrays containing input data, start maps, goal maps, and outputs.
    """
    all_x, all_s_map, all_g_map, all_y = [], [], [], []

    for dir_path in directories:
        try:
            x = np.loadtxt(os.path.join(dir_path, 'inputs.dat'))
            s_map = np.loadtxt(os.path.join(dir_path, 's_maps.dat'))
            g_map = np.loadtxt(os.path.join(dir_path, 'g_maps.dat'))
            y = np.loadtxt(os.path.join(dir_path, 'outputs.dat'))

            all_x.append(x)
            all_s_map.append(s_map)
            all_g_map.append(g_map)
            all_y.append(y)

        except FileNotFoundError as e:
            print(f"File not found in directory {dir_path}: {e}")
        except Exception as e:
            print(f"An error occurred in directory {dir_path}: {e}")

    # Concatenate all data into single arrays
    x_combined = np.concatenate(all_x, axis=0)
    s_map_combined = np.concatenate(all_s_map, axis=0)
    g_map_combined = np.concatenate(all_g_map, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    return x_combined, s_map_combined, g_map_combined, y_combined


def load_data(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load input data, start maps, goal maps, and outputs from the given directory.

    Args:
    input_path: Path to the directory containing data files.

    Returns:
    Four NumPy arrays containing input data, start maps, goal maps, and outputs.
    """
    x = np.loadtxt(os.path.join(input_path, 'inputs.dat'))
    s_map = np.loadtxt(os.path.join(input_path, 's_maps.dat'))
    g_map = np.loadtxt(os.path.join(input_path, 'g_maps.dat'))
    y = np.loadtxt(os.path.join(input_path, 'outputs.dat'))

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

def train_model(model: Model, x_train: np.ndarray, y_train: np.ndarray) -> keras.callbacks.History:
    """
    Train the neural network model.

    Args:
    model: Compiled neural network model.
    x_train: Training input data.
    y_train: Training output data.

    Returns:
    History object containing training metrics.
    """
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0, patience=10, verbose=1)
    save_weights = ModelCheckpoint(filepath='trained_model/100*100/weights_2d.keras', monitor='val_accuracy', verbose=1, save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=32, validation_split=1/14, epochs=EPOCHS, verbose=1, callbacks=[early_stop, save_weights])
    return history


def evaluate_model(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate the trained neural network model.

    Args:
    model: Trained neural network model.
    x_test: Testing input data.
    y_test: Testing output data.

    Returns:
    Test loss and accuracy.
    """
    score = model.evaluate(x_test, y_test, verbose=1)
    return score[0], score[1]


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
    input_path = directories_path

    print('Load data ...')
    x, s_map, g_map, y = load_data_from_directories(input_path)
    x3d, y = preprocess_data(x, s_map, g_map, y)

    x_train, y_train, x_test, y_test = split_data(x3d, y, train_ratio=TRAIN_RATIO)
    model = build_model(input_shape=x_train.shape[1:])
    model.summary()

    print('Train network ...')
    history = train_model(model, x_train, y_train)

    print('Save trained model ...')

    save_model(model, "trained_model/100*100/model_2d.keras")

    print('Test network ...')
    model = load_model("trained_model/100*100/model_2d.keras")
    loss, accuracy = evaluate_model(model, x_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    tf.keras.backend.clear_session()

