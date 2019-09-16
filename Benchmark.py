import csv
import sys
import time
import argparse
import builtins
import numpy as np
# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
from torch.nn import CrossEntropyLoss
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from PiNet import PiNet


def info(*msg):
    old_print(*msg, file=sys.stderr)


old_print = builtins.print
builtins.print = info


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


def get_splits():
    return StratifiedKFold(n_splits=10).split(np.zeros([len(dataset), 1]), dataset.data.y)


def get_args():
    parser = argparse.ArgumentParser(description='10-fold Cross Validation.')
    parser.add_argument('--dataset', dest='dataset_name', type=str,
                        default='MUTAG', required=False,
                        help='dataset name (default: MUTAG)')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=500, required=False,
                        help='sum the integers (default: 500)')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, required=False,
                        help='learning rate (default: 0.001)')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args.__dict__)

    writer = csv.writer(sys.stdout)

    models = [
        {
            'class': PiNet,
            'params': {
                'message_passing': 'GAT',
                'GAT_heads': [h1, h2],
                'dims': [d1, d2],
                'skip': skip
            },
        } for skip in [True, False] for h1 in [3, 5] for h2 in [3, 5] for d1 in [10, 15] for d2 in [10, 15]
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on {device}')
    writer.writerow(['dataset_name',
                     'model',
                     'model_params',
                     'split',
                     'epoch',
                     'time_for_epoch(ms)',
                     'train_loss',
                     'train_acc', 'test_acc',
                     'train_conf', 'test_conf'])

    for model_dict in models:

        dataset = TUDataset(root=f'data/{args.dataset_name}', name=args.dataset_name).shuffle()

        for split, (all_train_idx, test_idx) in enumerate(get_splits()):
            print(args.dataset_name, model_dict['class'].__name__, split)

            model = model_dict['class'](num_feats=dataset.num_features, num_classes=dataset.num_classes,
                                        **model_dict['params']).to(
                device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
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

            for epoch in range(args.epochs):
                start = time.time()
                train_loss = train()
                time_for_epoch = (time.time() - start) * 1e3
                train_acc, train_conf = evaluate(train_loader)
                # val_acc, val_conf = evaluate(val_loader)
                test_acc, test_conf = evaluate(test_loader)

                writer.writerow([args.dataset_name,
                                 model_dict["class"].__name__,
                                 model_dict['params'],
                                 split,
                                 epoch,
                                 time_for_epoch,
                                 train_loss,
                                 train_acc, test_acc,
                                 train_conf, test_conf])
