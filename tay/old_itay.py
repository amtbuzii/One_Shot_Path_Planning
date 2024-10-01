# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:39:20 2024
Path@Glance
@author: itayna
"""

# from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2

# @jit(nopython=True, parallel=True) # Set "nopython" mode for best performance, equivalent to @njit


# %% Load DB. If distances already calculated, just load them.
N = 30  # map size

data = np.load('30x30Inbal1DB.npz')
inp = data['inp']
target = data['target']
db_size = inp.shape[0]
del data

# %% Build DB - run this only once to calculate distances
'''
def load_data():
    x = np.loadtxt('from_Inbal/start_points_30x30_200000.dat')
    s_map = np.loadtxt('from_Inbal/end_point_mask_30x30_200000.dat')
    g_map = np.loadtxt('from_Inbal/path_mask_30x30_200000.dat')
    y = np.loadtxt('from_Inbal/polygon_mask30x30_200000.dat')
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

x, s_map, g_map, y = load_data()
x3d, y = preprocess_data(x, s_map, g_map, y)
del x, s_map, g_map 


def create_obs_scenario(ind): # Run this in order to add the gradients to 'ind' scenario
    obs = y[ind,:,:]
    to = x3d[ind,:,:,1]
    # row_max_idx = np.argmax(to, axis=0)  # Returns an array with column indices of row-wise maxima
    # col_max_idx = np.argmax(to, axis=1)  # Returns an array with row indices of column-wise maxima
    flat_max_idx = np.argmax(to)
    row_idx = flat_max_idx // to.shape[1]  # Integer division to get row index
    col_idx = flat_max_idx % to.shape[1]  # Modulo to get column index within the row
    to = (row_idx, col_idx)

    dst, vst = dijkstra_distances(to, obs)
    dst[dst == -1] = np.max(dst) + 2 # For not going through obstacles
    frm = x3d[ind,:,:,0]
    # row_max_idx = np.argmax(frm, axis=0)  # Returns an array with column indices of row-wise maxima
    # col_max_idx = np.argmax(frm, axis=1)  # Returns an array with row indices of column-wise maxima
    flat_max_idx = np.argmax(frm)
    row_idx = flat_max_idx // frm.shape[1]  # Integer division to get row index
    col_idx = flat_max_idx % frm.shape[1]  # Modulo to get column index within the row
    frm = (row_idx, col_idx)

    return obs, frm, to, dst, vst


def dijkstra_distances(to, obs): # This is probably NOT dijkstra. implement something better.
    vst = np.zeros((N,N))
    dst = np.zeros((N,N)) -1
    vst[to] = 1
    dst[to] = 0
    vst_num = 0
    vst_hist = 1
    stp = 0
    while vst_num - vst_hist != 0: # As long as there are unvisited points
        stp += 1
        # print(stp)
        vst_hist = np.sum(vst)
        for row in np.arange(1, N-1):
            for col in np.arange(1, N-1):
                if vst[row,col] == 1:
                    for drow in [-1, 0, 1]:
                        for dcol in [-1, 0, 1]:
                            if obs[row + drow, col + dcol] == 0:
                            # if vst[row + drow, col + dcol] == 0 and obs[row + drow, col + dcol] == 0:
                                vst[row + drow, col + dcol] = 1
                                if dst[row + drow, col + dcol] == -1:
                                    dst[row + drow, col + dcol] = dst[row, col] + (drow**2 + dcol**2)**0.5
                                else:
                                    dst[row + drow, col + dcol] = np.min((dst[row + drow, col + dcol], dst[row, col] + (drow**2 + dcol**2)**0.5))
        vst_num = np.sum(vst)
    return dst, vst



db_size = y.shape[0]
inp = np.zeros((db_size, N, N, 2))
target = np.zeros((db_size, N, N))

for ind in range(db_size):
    obs, frm, to, dst, vst = create_obs_scenario(ind)
    inp[ind,:,:,0] = obs
    inp[ind, to[0], to[1], 1] = 1    
    target[ind,:,:] = dst
    print(ind/db_size)
'''

# %% Define the model

model = keras.Sequential()
f = 64
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, activation="relu", padding='same', input_shape=(N, N, 2)))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', activation="relu"))
model.add(keras.layers.UpSampling2D(size=(2, 2)))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=f, kernel_size=5, padding='same', activation="relu"))
model.add(keras.layers.Conv2D(filters=1, kernel_size=5,
                              padding='same'))  # Adjust activation for image output (e.g., sigmoid for 0-1 values)

# Compile the model
model.compile(loss="mse", optimizer="adam")
model.summary()

loss = []
val_loss = []

# %% Train the model

mini_batch_size = 10
batch_size = 200
for epc in range(1000):

    # Augment samples to enrich DB
    tur_ang = np.random.randint(4)
    aug_inp = np.rot90(inp, tur_ang, axes=(1, 2))
    aug_target = np.rot90(target, tur_ang, axes=(1, 2))
    if np.random.rand() < 0.5:
        aug_inp = np.fliplr(aug_inp)
        aug_target = np.fliplr(aug_target)

    frm_ind = np.random.randint(db_size - batch_size * 2)
    hist = model.fit(aug_inp[frm_ind:frm_ind + batch_size], aug_target[frm_ind:frm_ind + batch_size], epochs=1,
                     batch_size=mini_batch_size,
                     validation_data=(inp[db_size - batch_size:-1], target[db_size - batch_size:-1]))
    loss.append(hist.history['loss'])  # calculated on all but last batch_size
    val_loss.append(hist.history['val_loss'])  # calculated on last batch_size

    if np.random.rand() > 0.95:
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.semilogy(loss)
        plt.semilogy(val_loss)
        plt.grid(True)
        plt.title('Train / Val Loss')
        rnd = np.random.randint(0, db_size - 2)
        plt.subplot(2, 2, 2)
        plt.title('Input')
        plt.imshow(inp[rnd, :, :, 0] + inp[rnd, :, :, 1] * 0.5)
        plt.subplot(2, 2, 3)
        plt.title('Target')
        plt.imshow(target[rnd])
        plt.subplot(2, 2, 4)
        plt.title('Prediction')
        plt.imshow(model.predict(inp[rnd:rnd + 1])[0, :, :, 0])
        plt.draw();
        plt.pause(0.1)






