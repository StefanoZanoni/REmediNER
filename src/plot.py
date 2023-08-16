import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi=100)


def plot_heat_map(cm, annotation):
    if not os.path.exists(f'./NER/plots'):
        os.makedirs(f'./NER/plots', exist_ok=True)
    if not os.path.exists(f'./NER/plots/Heat maps'):
        os.makedirs(f'./NER/plots/Heat maps', exist_ok=True)
    labels = ['O', 'B-Drug', 'I-Drug', 'B-Effect', 'I-Effect']

    train_cm = pd.DataFrame(train_cm, index=labels, columns=labels)
    train_heatmap = sns.heatmap(train_cm, annot=True)
    train_fig = train_heatmap.get_figure()
    train_fig.savefig(f'./NER/plots/Heat maps/{annotation}')
    plt.clf()
