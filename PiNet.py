import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import accuracy_score
from torch.nn import Module
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, to_dense_batch


class PiNet(Module):

    def __init__(self, *dims):
        super(PiNet, self).__init__()
        self.dims = dims
        self.gcn_a1 = GCNConv(dims[0], dims[1])
        self.gcn_a2 = GCNConv(dims[1], dims[2])
        self.gcn_x1 = GCNConv(dims[0], dims[1])
        self.gcn_x2 = GCNConv(dims[1], dims[2])
        self.linear2 = torch.nn.Linear(dims[2] ** 2, dims[-1])

    def forward(self, data):
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

        a_1 = F.relu(self.gcn_a1(x, edge_index))
        a_2 = softmax(self.gcn_a2(a_1, edge_index), batch)
        x_1 = F.relu(self.gcn_x1(x, edge_index))
        x_2 = F.relu(self.gcn_x2(x_1, edge_index))

        a_batch, _ = to_dense_batch(a_2, batch)
        a_t = a_batch.transpose(2, 1)
        x_batch, _ = to_dense_batch(x_2, batch)
        prods = torch.bmm(a_t, x_batch)
        flat = torch.flatten(prods, 1, -1)
        batch_out = self.linear2(flat)

        final = F.softmax(batch_out, dim=-1)

        return final


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

    return accuracy_score(labels, predictions), metrics.confusion_matrix(labels, predictions)


dataset = TUDataset(root='data/mutag', name='MUTAG').shuffle()
# device = torch.device("cuda:0")
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PiNet(dataset.num_features, 100, 100, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
crit = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(dataset[:152], batch_size=152)
val_loader = DataLoader(dataset[152:170], batch_size=18)
test_loader = DataLoader(dataset[170:], batch_size=18)

conf = None
for epoch in range(200):
    loss = train()
    train_acc, train_conf = evaluate(train_loader)
    val_acc, val_conf = evaluate(val_loader)
    test_acc, test_conf = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))

print('train\n', train_conf)
print('test\n', test_conf)