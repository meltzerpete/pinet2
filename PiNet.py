import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax, to_dense_batch


class PiNet(Module):

    def __init__(self, num_feats=None, dims=None, num_classes=2, message_passing='GCN', GCN_improved=False,
                 GAT_heads=None):
        super(PiNet, self).__init__()
        if dims is None:
            dims = [64, 64]
        if message_passing == 'GCN':
            from torch_geometric.nn import GCNConv
            self.mp_a1 = GCNConv(num_feats, dims[0], improved=GCN_improved)
            self.mp_a2 = GCNConv(dims[0], dims[1], improved=GCN_improved)
            self.mp_x1 = GCNConv(num_feats, dims[0], improved=GCN_improved)
            self.mp_x2 = GCNConv(dims[0], dims[1], improved=GCN_improved)
            self.linear2 = torch.nn.Linear(dims[1] ** 2, num_classes)

        elif message_passing == 'GAT':
            if (GAT_heads == None):
                GAT_heads = [5, 2]
            self.mp_a1 = GATConv(num_feats, dims[0], heads=GAT_heads[0])
            self.mp_a2 = GATConv(dims[0] * GAT_heads[0], dims[1], heads=GAT_heads[1])
            self.mp_x1 = GATConv(num_feats, dims[0], heads=GAT_heads[0])
            self.mp_x2 = GATConv(dims[0] * GAT_heads[0], dims[1], heads=GAT_heads[1])
            self.linear2 = torch.nn.Linear((dims[1] * GAT_heads[1]) ** 2, num_classes)

    def forward(self, data):
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

        a_1 = F.relu(self.mp_a1(x, edge_index))
        a_2 = softmax(self.mp_a2(a_1, edge_index), batch)
        x_1 = F.relu(self.mp_x1(x, edge_index))
        x_2 = F.relu(self.mp_x2(x_1, edge_index))

        a_batch, _ = to_dense_batch(a_2, batch)
        a_t = a_batch.transpose(2, 1)
        x_batch, _ = to_dense_batch(x_2, batch)
        prods = torch.bmm(a_t, x_batch)
        flat = torch.flatten(prods, 1, -1)
        batch_out = self.linear2(flat)

        final = F.softmax(batch_out, dim=-1)

        return final
