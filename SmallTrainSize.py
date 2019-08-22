import json
import math
import os.path
import csv
import time
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.nn import CrossEntropyLoss
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from PiNet import PiNet


def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions).argmax(axis=1)

    return accuracy_score(labels, predictions), metrics.confusion_matrix(labels, predictions).ravel()


def get_experiment_count(file):
    if os.path.isfile(file):
        count = json.load(open(file, 'r')) + 1
    else:
        count = 0
    json.dump(count, open(file, 'w'))
    return count


def get_splits(train_size):
    return StratifiedShuffleSplit(n_splits=10, train_size=train_size).split(np.zeros([len(dataset), 1]), dataset.data.y)


if __name__ == '__main__':
    experiment_count = get_experiment_count('small_train.count')

    log_file = open('small_train.log', 'a')
    writer = csv.writer(log_file)

    # datasets = ['MUTAG', 'PTC_MM', 'PTC_MR', 'PTC_FM', 'PTC_FR']
    datasets = ['PROTEINS']
    models = [
        # {
        #     'class': PiNet,
        #     'params': {
        #         'message_passing': 'GCN',
        #         'GCN_improved': True
        #     }
        # },
        {
            'class': PiNet,
            'params': {
                'message_passing': 'GCN',
                'GCN_improved': False
            },
        },
        # {
        #     'class': PiNet,
        #     'params': {
        #         'message_passing': 'GAT',
        #         'GAT_heads': [2, 2]
        #     },
        # },
        # {
        #     'class': PiNet,
        #     'params': {
        #         'message_passing': 'GAT',
        #         'GAT_heads': [3, 2]
        #     },
        # },
    ]
    # train_sizes = [4, 8, 12, 16, 20, 24, 28, 32]
    train_sizes = [10, 20, 30, 40, 50, 60]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'exp: {experiment_count}, running on {device}')
    writer.writerow([experiment_count,
                     'dataset_name',
                     'model',
                     'model_params',
                     'train_size',
                     'split',
                     'epoch',
                     'time_for_epoch(ms)',
                     'train_loss',
                     'train_acc', 'test_acc',
                     'train_conf', 'test_conf'])
    log_file.flush()

    for dataset_name in datasets:
        for model_dict in models:
            for train_size in train_sizes:

                dataset = TUDataset(root=f'data/{dataset_name}', name=dataset_name).shuffle()

                for split, (all_train_idx, test_idx) in enumerate(get_splits(train_size)):
                    print(dataset_name, model_dict['class'].__name__, split)

                    model = model_dict['class'](dataset.num_features, 64, 64, dataset.num_classes,
                                                **model_dict['params']).to(
                        device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
                    crit = CrossEntropyLoss()

                    # convert idx to torch tensors
                    train_idx = torch.tensor(all_train_idx, dtype=torch.long)
                    test_idx = torch.tensor(test_idx, dtype=torch.long)

                    train_loader = DataLoader(dataset[train_idx], batch_size=len(train_idx))
                    test_loader = DataLoader(dataset[test_idx], batch_size=len(test_idx))

                    for epoch in range(300):
                        start = time.time()
                        train_loss = train()
                        time_for_epoch = (time.time() - start) * 1e3
                        train_acc, train_conf = evaluate(train_loader)
                        test_acc, test_conf = evaluate(test_loader)

                        writer.writerow([experiment_count,
                                         dataset_name,
                                         model_dict["class"].__name__,
                                         model_dict['params'],
                                         train_size,
                                         split,
                                         epoch,
                                         time_for_epoch,
                                         train_loss,
                                         train_acc, test_acc,
                                         train_conf, test_conf])
                        log_file.flush()
