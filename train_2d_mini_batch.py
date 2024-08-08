from typing import Tuple
import os
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

class SaveModelCallback(Callback):
    def __init__(self, filepath):
        super(SaveModelCallback, self).__init__()
        self.filepath = filepath

    def on_batch_end(self, batch, logs=None):
        self.model.save(self.filepath)

def train_model_on_batch(model: Model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
    """
    Train the neural network model on a batch of data.

    Args:
    model: Compiled neural network model.
    x_train: Training input data.
    y_train: Training output data.
    x_val: Validation input data.
    y_val: Validation output data.

    Returns:
    History object containing training metrics.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(filepath='trained_model/mini_batch/best_model.keras', monitor='val_loss', save_best_only=True),
        SaveModelCallback(filepath='trained_model/mini_batch/latest_model.keras')
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

    print('Train network ...')
    for directory in directories_path:
        print(f'Loading data from {directory} ...')
        x, s_map, g_map, y = load_data_batch(directory)
        x3d, y = preprocess_data(x, s_map, g_map, y)

        print('Split data into training and testing sets ...')
        x_train, x_test, y_train, y_test = train_test_split(x3d, y, test_size=1-TRAIN_RATIO, random_state=42)

        print('Train model on current batch ...')
        train_model_on_batch(model, x_train, y_train, x_test, y_test)

        print('Save model after current batch ...')
        save_model(model, "trained_model/mini_batch/model_2d.keras")

        print('Load the latest model ...')
        model = load_model("trained_model/mini_batch/model_2d.keras")

        print('Evaluate model on test set for current batch ...')
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        print(f'Test loss for current batch: {loss}')
        print(f'Test accuracy for current batch: {accuracy}')
        tf.keras.backend.clear_session()

    print('Test network on the last chunk ...')
    x_test, s_map_test, g_map_test, y_test = load_data_batch(directories_path[-1])
    x3d_test, y_test = preprocess_data(x_test, s_map_test, g_map_test, y_test)
    model = load_model("trained_model/100*100/model_2d.keras")
    loss, accuracy = model.evaluate(x3d_test, y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    tf.keras.backend.clear_session()

