import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('benchmark.log')

groups = df.groupby(['dataset_name', 'model_params', 'split'])

best = df.iloc[groups.idxmax()['val_acc']]

test_accs = best.groupby(['dataset_name', 'model_params'])['test_acc']

means = test_accs.mean().reset_index().groupby('dataset_name')
stds = test_accs.std().reset_index().groupby('dataset_name')

for group_key in means.groups:
    fig = plt.figure()
    plt.title(str(group_key))
    _means = means.get_group(group_key)
    _stds = stds.get_group(group_key)

    x_pos = np.arange(len(_means))
    plt.bar(x_pos, _means['test_acc'], yerr=_stds['test_acc'])

    plt.xticks(x_pos, _means['model_params'], rotation=30)

    plt.show()
