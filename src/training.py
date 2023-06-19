n_splits = 1


def training(data_x, data_y, model):
    # for train_index, val_index in k_fold(data_x, n_splits):
    #     train_x = [data_x[i] for i in train_index]
    #     train_y = [data_y[i] for i in train_index]
    #     val_x = [data_x[i] for i in val_index]
    #     val_y = [data_y[i] for i in val_index]
    history = model.fit([data_x[0], data_x[1]], [data_x[2], (data_y[0], data_y[1])],
                        epochs=1,
                        batch_size=1)
    print(history.history['train_loss'])
