import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import cv2
from typing import Tuple, List

# Constants
N = 100  # Map size
BATCH_SIZE = 200
MINI_BATCH_SIZE = 10
EPOCHS = 1000
FILTERS = 64


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load input and target data from a .npz file.

    Parameters:
        filename (str): Path to the .npz file containing the data.

    Returns:
        tuple: inp, target arrays and the size of the dataset.
    """
    data = np.load(filename)
    inp, target = data['inp'], data['target']
    db_size = inp.shape[0]
    del data  # Clear unused data from memory
    return inp, target, db_size


def build_model(input_shape: Tuple[int, int, int], filters: int) -> Sequential:
    """
    Build and compile the convolutional neural network model.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (N, N, 2)).
        filters (int): Number of filters for the convolutional layers.

    Returns:
        model (keras.Sequential): Compiled Keras model.
    """
    model = Sequential()
    for _ in range(12):
        model.add(Conv2D(filters=filters, kernel_size=5, activation="relu", padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for _ in range(3):
        model.add(Conv2D(filters=filters, kernel_size=3, activation="relu", padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    for _ in range(3):
        model.add(Conv2D(filters=filters, kernel_size=5, activation="relu", padding='same'))
    model.add(Conv2D(filters=1, kernel_size=5, padding='same'))  # Output layer without activation for raw pixel values

    model.compile(loss="mse", optimizer="adam")
    return model


def augment_data(inp: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment input and target data by random rotations and flipping.

    Parameters:
        inp (numpy.ndarray): Input data to augment.
        target (numpy.ndarray): Target data to augment.

    Returns:
        tuple: Augmented input and target arrays.
    """
    # Random rotation by multiples of 90 degrees
    rot_angle = np.random.randint(4)
    aug_inp = np.rot90(inp, rot_angle, axes=(1, 2))
    aug_target = np.rot90(target, rot_angle, axes=(1, 2))

    # Random horizontal flip
    if np.random.rand() < 0.5:
        aug_inp = np.fliplr(aug_inp)
        aug_target = np.fliplr(aug_target)

    return aug_inp, aug_target


def plot_results(loss: List[float], val_loss: List[float], inp: np.ndarray, target: np.ndarray, model: Sequential,
                 db_size: int) -> None:
    """
    Visualize training loss and compare input, target, and predicted outputs.

    Parameters:
        loss (list): List of training loss values.
        val_loss (list): List of validation loss values.
        inp (numpy.ndarray): Input dataset.
        target (numpy.ndarray): Target dataset.
        model (keras.Sequential): Trained Keras model.
        db_size (int): Size of the dataset.
    """
    plt.clf()

    # Plot the loss and validation loss
    plt.subplot(2, 2, 1)
    plt.semilogy(loss, label='Train Loss')
    plt.semilogy(val_loss, label='Val Loss')
    plt.grid(True)
    plt.legend()
    plt.title('Train / Val Loss')

    # Select random sample for visualization
    rnd = np.random.randint(0, db_size - 2)

    # Plot Input
    plt.subplot(2, 2, 2)
    plt.title('Input')
    plt.imshow(inp[rnd, :, :, 0] + inp[rnd, :, :, 1] * 0.5, cmap='gray')

    # Plot Target
    plt.subplot(2, 2, 3)
    plt.title('Target')
    plt.imshow(target[rnd], cmap='gray')

    # Plot Prediction
    plt.subplot(2, 2, 4)
    plt.title('Prediction')
    predicted = model.predict(inp[rnd:rnd + 1])[0, :, :, 0]
    plt.imshow(predicted, cmap='gray')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


def train_model(model: Sequential, inp: np.ndarray, target: np.ndarray, db_size: int, epochs: int, batch_size: int,
                mini_batch_size: int) -> None:
    """
    Train the model using augmented data and plot progress during training.

    Parameters:
        model (keras.Sequential): Keras model to train.
        inp (numpy.ndarray): Input data for training.
        target (numpy.ndarray): Target data for training.
        db_size (int): Size of the dataset.
        epochs (int): Number of epochs for training.
        batch_size (int): Size of the batches for training.
        mini_batch_size (int): Mini-batch size for training.
    """
    loss, val_loss = [], []

    for epc in range(epochs):
        # Augment the input and target data
        aug_inp, aug_target = augment_data(inp, target)

        # Select a random batch
        frm_ind = np.random.randint(db_size - batch_size * 2)

        # Train the model on the augmented batch
        hist = model.fit(aug_inp[frm_ind:frm_ind + batch_size], aug_target[frm_ind:frm_ind + batch_size], epochs=1,
                         batch_size=mini_batch_size,
                         validation_data=(inp[db_size - batch_size:-1], target[db_size - batch_size:-1]))

        # Append loss and validation loss
        loss.append(hist.history['loss'])
        val_loss.append(hist.history['val_loss'])

        # Occasionally plot results during training
        if np.random.rand() > 0.95:
            plot_results(loss, val_loss, inp, target, model, db_size)


if __name__ == "__main__":
    # Load data
    inp, target, db_size = load_data('amit_compressed.npz')

    # Build and compile model
    model = build_model((N, N, 2), FILTERS)
    model.summary()

    # Train the model
    train_model(model, inp, target, db_size, EPOCHS, BATCH_SIZE, MINI_BATCH_SIZE)
