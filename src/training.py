from src.data_utilities import k_fold

import keras

n_splits = 10


def training(data_x, data_y, model):
    for train_index, val_index in k_fold(data_x, data_y, n_splits):
        train_x, train_y, val_x, val_y = data_x[train_index], data_y[train_index], val_x[val_index], val_y[val_index]
        model.fit(train_x, train_y,
                  epochs=10,
                  batch_size=2,
                  validation_data=(val_x, val_y))
