import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi=100)


def plot_heat_map(cm, annotation):
    if not os.path.exists('../plots/ner_plots'):
        os.makedirs('../plots/ner_plots', exist_ok=True)
    if not os.path.exists('../plots/ner_plots/Heat maps'):
        os.makedirs('../plots/ner_plots/Heat maps', exist_ok=True)
    labels = ['O', 'B-Drug', 'I-Drug', 'B-Effect', 'I-Effect']

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    train_heatmap = sns.heatmap(cm, annot=True)
    train_fig = train_heatmap.get_figure()
    train_fig.savefig(f'../plots/ner_plots/Heat maps/{annotation}')
    plt.clf()
