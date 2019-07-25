import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import Module
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, softmax

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
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

        edge_index, _ = add_self_loops(edge_index)

        a_1 = F.relu(self.gcn_a1(x, edge_index))
        a_2 = self.gcn_a2(a_1, edge_index)
        x_1 = F.relu(self.gcn_x1(x, edge_index))
        x_2 = F.relu(self.gcn_x2(x_1, edge_index))

        out = []
        for g in range(num_graphs):
            x_ = x_2[~(batch == g)]
            a_ = a_2[~(batch == g)]

            a = F.softmax(torch.transpose(a_, 1, 0), -1)
            h = torch.matmul(a, x_)

            o = self.linear2(h.reshape(1, -1))
            out.append(o)
        batch_out = torch.cat(out, 0)
        # print(batch_out[:2])

        final = F.softmax(batch_out, dim=-1)

        # print(final[0])
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

    return accuracy_score(labels, predictions)


dataset = TUDataset(root='data/mutag', name='MUTAG').shuffle()
# device = torch.device("cuda:0")
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PiNet(dataset.num_features, 32, 32, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)
crit = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(dataset[:152], batch_size=152)
val_loader = DataLoader(dataset[152:170], batch_size=18)
test_loader = DataLoader(dataset[170:], batch_size=18)

for epoch in range(1000):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))
