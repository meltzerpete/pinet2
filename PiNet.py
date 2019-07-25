import torch
from torch.nn import Module
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool


class PiNet(Module):

    def __init__(self, *dims):
        super(PiNet, self).__init__()
        self.dims = dims
        self.linear1 = torch.nn.Linear(dims[0], dims[1])
        self.linear2 = torch.nn.Linear(dims[1], dims[2])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = torch.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.softmax(global_mean_pool(o, batch), -1)


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
    return loss_all / len(train_dataset)


dataset = TUDataset(root='data/proteins', name='PROTEINS')
device = torch.device('cpu')
model = PiNet(dataset.num_features, 50, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.CrossEntropyLoss()
train_dataset = dataset
train_loader = DataLoader(train_dataset, batch_size=1)

for epoch in range(100):
    print(f'epoch: {epoch}')
    train()
