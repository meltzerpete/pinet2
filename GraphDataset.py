import os
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, InMemoryDataset


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{name}.txt' for name in
                ['graph_indicator', 'A', 'graph_labels', 'node_labels', 'edge_labels']]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def raw_file(self, name):
        return os.path.join(self.raw_dir, name) + '.txt'

    def process(self):
        # Read data into huge `Data` list.
        graph_ind = np.loadtxt(self.raw_file('graph_indicator'), dtype='int')
        all_edges = np.loadtxt(self.raw_file('A'), delimiter=',')
        all_x = np.loadtxt(self.raw_file('node_labels')).reshape(-1, 1)
        all_edge_labels = np.loadtxt(self.raw_file('edge_labels')).reshape(-1, 1)
        all_y = np.loadtxt(self.raw_file('graph_labels')).reshape(-1, 1)

        enc_x = OneHotEncoder(categories='auto').fit(all_x)
        enc_edge_labels = OneHotEncoder(categories='auto').fit(all_edge_labels)
        enc_y = OneHotEncoder(categories='auto').fit(all_y)

        data_list = []
        for i in range(graph_ind.max()):
            idx = np.argwhere(graph_ind == i + 1).flatten()

            x = enc_x.transform(all_x[idx]).todense()
            edge_index = np.transpose(all_edges[idx])
            edge_attr = enc_edge_labels.transform(all_edge_labels[idx]).todense()
            y = enc_y.transform(all_y[i].reshape(1, -1)).todense()
            data = Data(x=torch.tensor(x),
                        edge_index=torch.tensor(edge_index),
                        edge_attr=torch.tensor(edge_attr),
                        y=torch.tensor(y))

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
