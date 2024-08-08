from typing import Tuple
import keras
import os
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split


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

    model = Model(inputs=x, outputs=net)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

def train_model_on_batch(model: Model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
    """
    Train the neural network model on a batch of data.

    Args:
    model: Compiled neural network model.
    x_batch: Batch input data.
    y_batch: Batch output data.

    Returns:
    History object containing training metrics.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)
    return history


def save_model(model: Model, filename: str) -> None:
    """
    Save the trained neural network model to a file.

    Args:
    model: Trained neural network model.
    filename: Name of the file to save the model.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    model.save(filename)


if __name__ == "__main__":
    print('Build model ...')
    model = build_model(input_shape=(N, N, 3))
    model.summary()

    print('Load and preprocess data ...')
    x_list, s_map_list, g_map_list, y_list = [], [], [], []
    for directory in directories_path:
        x, s_map, g_map, y = load_data_batch(directory)
        x_list.append(x)
        s_map_list.append(s_map)
        g_map_list.append(g_map)
        y_list.append(y)

    x = np.concatenate(x_list, axis=0)
    s_map = np.concatenate(s_map_list, axis=0)
    g_map = np.concatenate(g_map_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    x3d, y = preprocess_data(x, s_map, g_map, y)

    print('Split data into training and testing sets ...')
    x_train, x_test, y_train, y_test = train_test_split(x3d, y, test_size=1-TRAIN_RATIO, random_state=42)

    print('Train model ...')
    train_model_on_batch(model, x_train, y_train, x_test, y_test)

    print('Save trained model ...')
    save_model(model, "trained_model/100*100/model_2d.keras")

    print('Load best model and evaluate ...')
    best_model = load_model('best_model.keras')
    loss, accuracy = best_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    tf.keras.backend.clear_session()
