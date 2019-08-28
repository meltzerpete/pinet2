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


def get_splits():
    return StratifiedKFold(n_splits=10).split(np.zeros([len(dataset), 1]), dataset.data.y)


if __name__ == '__main__':
    experiment_count = get_experiment_count('benchmark.count')
    print(f'experiment count: {experiment_count}')

    log_file = open('benchmark.log', 'a')
    writer = csv.writer(log_file)

    datasets = ['MUTAG', 'PTC_MM', 'PTC_MR', 'PTC_FM', 'PTC_FR', 'NCI1', 'NCI109', 'PROTEINS']
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
                'GCN_improved': False,
                'dims': [32, 32],
            },
        },
        {
            'class': PiNet,
            'params': {
                'message_passing': 'GCN',
                'GCN_improved': False,
                'dims': [64, 32],
            },
        },
        {
            'class': PiNet,
            'params': {
                'message_passing': 'GCN',
                'GCN_improved': False,
                'dims': [32, 64],
            },
        },
        {
            'class': PiNet,
            'params': {
                'message_passing': 'GCN',
                'GCN_improved': False,
                'dims': [64, 64],
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on {device}')
    writer.writerow([experiment_count,
                     'dataset_name',
                     'model',
                     'model_params',
                     'split',
                     'epoch',
                     'time_for_epoch(ms)',
                     'train_loss',
                     'train_acc', 'test_acc',
                     'train_conf', 'test_conf'])
    log_file.flush()

    for dataset_name in datasets:
        for model_dict in models:

            dataset = TUDataset(root=f'data/{dataset_name}', name=dataset_name).shuffle()

            for split, (all_train_idx, test_idx) in enumerate(get_splits()):
                print(dataset_name, model_dict['class'].__name__, split)

                model = model_dict['class'](num_feats=dataset.num_features, num_classes=dataset.num_classes,
                                            **model_dict['params']).to(
                    device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
                crit = CrossEntropyLoss()

                # convert idx to torch tensors
                train_idx = torch.tensor(all_train_idx, dtype=torch.long)
                test_idx = torch.tensor(test_idx, dtype=torch.long)

                # train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1)
                #                           .split(np.zeros([len(all_train_idx), 1]), dataset.data.y[all_train_idx]))

                # train_idx = torch.tensor(train_idx, dtype=torch.long)
                # val_idx = torch.tensor(val_idx, dtype=torch.long)

                train_loader = DataLoader(dataset[train_idx], batch_size=len(train_idx))
                # val_loader = DataLoader(dataset[val_idx], batch_size=len(val_idx))
                test_loader = DataLoader(dataset[test_idx], batch_size=len(test_idx))

                for epoch in range(300):
                    start = time.time()
                    train_loss = train()
                    time_for_epoch = (time.time() - start) * 1e3
                    train_acc, train_conf = evaluate(train_loader)
                    # val_acc, val_conf = evaluate(val_loader)
                    test_acc, test_conf = evaluate(test_loader)

                    writer.writerow([experiment_count,
                                     dataset_name,
                                     model_dict["class"].__name__,
                                     model_dict['params'],
                                     split,
                                     epoch,
                                     time_for_epoch,
                                     train_loss,
                                     train_acc, test_acc,
                                     train_conf, test_conf])
                    log_file.flush()
