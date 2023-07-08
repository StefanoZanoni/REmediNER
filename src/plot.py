import os

import numpy as np

from matplotlib import pyplot as plt


def plot_loss(training_data, validation_data, fold, task):

    if not os.path.exists(f'./{task}/plots'):
        os.makedirs(f'./{task}/plots')
    if not os.path.exists(f'./{task}/plots/Losses'):
        os.makedirs(f'./{task}/plots/Losses')

    num_epochs = np.arange(1, len(training_data) + 1)
    plt.plot(num_epochs, training_data, 'g', label='Training loss')
    plt.plot(num_epochs, validation_data, 'b', label='Validation loss')
    plt.title(f'Training and Validation loss fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{task}/plots/Losses/Loss-Fold{fold}.png')


def plot_metrics(training_metrics, validation_metrics, fold):

    if not os.path.exists('./NER/plots/Metrics'):
        os.makedirs('./NER/plots/Metrics')

    num_epochs = np.arange(1, len(training_metrics) + 1)

    train_entities = {key: {'f1-score': [], 'precision': [], 'recall': []} for key in training_metrics[0]}
    for i in range(len(num_epochs)):
        for key in training_metrics[i]:
            for metric in training_metrics[i][key]:
                if metric != 'support':
                    train_entities[key][metric].append(training_metrics[i][key][metric])

    val_entities = {key: {'f1-score': [], 'precision': [], 'recall': []} for key in validation_metrics[0]}
    for i in range(len(num_epochs)):
        for key in validation_metrics[i]:
            for metric in validation_metrics[i][key]:
                if metric != 'support':
                    val_entities[key][metric].append(validation_metrics[i][key][metric])

    for key in train_entities:
        if not os.path.exists(f'./NER/plots/Metrics/{key}'):
            os.makedirs(f'./NER/plots/Metrics/{key}')
        f1_train, f1_val = None, None
        precision_train, precision_val = None, None
        recall_train, recall_val = None, None
        for metric in train_entities[key]:
            if metric == 'f1-score':
                f1_train = train_entities[key][metric]
                f1_val = val_entities[key][metric]
            elif metric == 'precision':
                precision_train = train_entities[key][metric]
                precision_val = val_entities[key][metric]
            elif metric == 'recall':
                recall_train = train_entities[key][metric]
                recall_val = val_entities[key][metric]
        plt.plot(num_epochs, f1_train, 'r', label='Training F1-score')
        plt.plot(num_epochs, f1_val, 'r--', label='Validation F1-score')
        plt.plot(num_epochs, precision_train, 'g', label='Training precision')
        plt.plot(num_epochs, precision_val, 'g--', label='Validation precision')
        plt.plot(num_epochs, recall_train, 'b', label='Training recall')
        plt.plot(num_epochs, recall_val, 'b--', label='Validation recall')
        plt.title(f'Training and Validation {key} metrics on fold {fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.savefig(f'./NER/plots/Metrics/{key}/ Metrics-Fold{fold}.png')
        plt.clf()

# def plot_test(test_data):
#     num_epochs = np.arange(1, len(training_data) + 1)
#     plt.plot(num_epochs, training_data, 'g', label='Training loss')
#     plt.title('Test loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()