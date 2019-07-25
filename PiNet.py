import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import Module
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv


class PiNet(Module):

    def __init__(self, *dims):
        super(PiNet, self).__init__()
        self.dims = dims
        self.gcn_a1 = GCNConv(dims[0], dims[1])
        self.gcn_a2 = GCNConv(dims[1], dims[2])
        self.gcn_x1 = GCNConv(dims[0], dims[1])
        self.gcn_x2 = GCNConv(dims[1], dims[2])
        self.linear2 = torch.nn.Linear(dims[2] * dims[2], dims[-1])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        a_1 = self.gcn_a1(x, edge_index)
        a_2 = self.gcn_a2(a_1, edge_index)

        x_1 = self.gcn_a1(x, edge_index)
        x_2 = self.gcn_a2(x_1, edge_index)

        a = torch.transpose(a_2, 1, 0)

        h = torch.matmul(a, x_2)

        o = self.linear2(h.flatten().reshape(1, -1))
        return F.softmax(o, dim=-1)


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

    return accuracy_score(labels, predictions)


dataset = TUDataset(root='data/mutag', name='MUTAG').shuffle()
# device = torch.device("cuda:0")
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PiNet(dataset.num_features, 10, 10, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(dataset[:100], batch_size=2)
val_loader = DataLoader(dataset[100:144], batch_size=2)
test_loader = DataLoader(dataset[144:], batch_size=2)

for epoch in range(1000):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))