import numpy as np

from matplotlib import pyplot as plt


def plot_loss(training_data, validation_data, fold):
    num_epochs = np.arange(1, len(training_data) + 1)
    plt.plot(num_epochs, training_data, 'g', label='Training loss')
    plt.plot(num_epochs, validation_data, 'b', label='Validation loss')
    plt.title(f'Training and Validation loss fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_metrics(train_precisions, train_recalls, train_f1s, validation_precisions, validation_recalls, validation_f1s,
                 fold):
    num_epochs = np.arange(1, len(train_precisions) + 1)
    plt.plot(num_epochs, train_precisions, 'r', label='Training precision')  # precision red
    plt.plot(num_epochs, train_recalls, 'g', label='Training recall')
    plt.plot(num_epochs, train_f1s, 'b', label='Training F1')
    plt.plot(num_epochs, validation_precisions, 'r--', label='Validation precision')
    plt.plot(num_epochs, validation_recalls, 'g--', label='Validation recall')
    plt.plot(num_epochs, validation_f1s, 'b--', label='Validation F1')
    plt.title(f'Training and Validation metrics fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

# def plot_test(test_data):
#     num_epochs = np.arange(1, len(training_data) + 1)
#     plt.plot(num_epochs, training_data, 'g', label='Training loss')
#     plt.title('Test loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
