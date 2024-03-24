import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

import numpy as np

from math import ceil
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.transforms import ToDense

NUM_SAGE_LAYERS = 3

from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_sort_pool
from torch_geometric.nn import aggr
from torch_geometric.utils import add_self_loops, degree


# Code from A fair comparison of GNN for graph classification 

#the data is from a DataLoader object of Pytorch Geometric (is of type Batch)
#data.x = node features
#data.edge_index = describes adjacency matrix in "COO format" (shape 2 x nb of edges)


# GIN

class GIN(torch.nn.Module):
    #there is no dense NN layers at the end. Instead we apply one linear layer after pooling. Should we switch to a NN layer?

    def __init__(self, dim_features, dim_target, 
                dropout=0.0, hidden_units=[32, 32], train_eps=False, aggregation='mean'):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.embeddings_dim = [hidden_units[0]] + hidden_units
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.dim_target=dim_target

        train_eps = train_eps
        if aggregation == 'sum':
            self.pooling = global_add_pool
        elif aggregation == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

        #new
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out


# DiffPool

class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = nn.Linear((NUM_SAGE_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            # GNN's intermediate representation is given by the concatenation of SAGE layers
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.conv3(x2, adj, mask)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x


class DiffPoolLayer(nn.Module):
    """
    Applies GraphSAGE convolutions and then performs pooling
    """
    def __init__(self, dim_input, dim_hidden, dim_embedding, no_new_clusters):
        """
        :param dim_input:
        :param dim_hidden: embedding size of first 2 SAGE convolutions
        :param dim_embedding: embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        :param no_new_clusters: number of clusters after pooling (eq. 6, dim of S)
        """
        super().__init__()
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPool(nn.Module):
    """
    Computes multiple DiffPoolLayers
    """
    def __init__(self, dim_features, dim_target, 
        max_num_nodes=1000, num_diffpool_layers=2, gnn_dim_hidden=64, dim_embedding=128, dim_embedding_MLP=50):
        super().__init__()

        self.max_num_nodes = max_num_nodes
        #num_diffpool_layers = config['num_layers']
        #gnn_dim_hidden = config['gnn_dim_hidden']  # embedding size of first 2 SAGE convolutions
        #dim_embedding = config['dim_embedding']  # embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        #dim_embedding_MLP = config['dim_embedding_MLP']  # hidden neurons of last 2 MLP layers

        self.num_diffpool_layers = num_diffpool_layers

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_diffpool_layers == 1 else 0.25

        gnn_dim_input = dim_features
        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        gnn_embed_dim_output = (NUM_SAGE_LAYERS - 1) * gnn_dim_hidden + dim_embedding

        layers = []
        for i in range(num_diffpool_layers):
            diffpool_layer = DiffPoolLayer(gnn_dim_input, gnn_dim_hidden, dim_embedding, no_new_clusters)
            layers.append(diffpool_layer)

            # Update embedding sizes
            gnn_dim_input = gnn_embed_dim_output
            no_new_clusters = ceil(no_new_clusters * coarse_factor)

        self.diffpool_layers = nn.ModuleList(layers)

        # After DiffPool layers, apply again layers of GraphSAGE convolutions
        self.final_embed = SAGEConvolutions(gnn_embed_dim_output, gnn_dim_hidden, dim_embedding, lin=False)
        final_embed_dim_output = gnn_embed_dim_output * (num_diffpool_layers + 1)

        self.lin1 = nn.Linear(final_embed_dim_output, dim_embedding_MLP)
        self.lin2 = nn.Linear(dim_embedding_MLP, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        # data = ToDense(data.num_nodes)(data)
        # TODO describe mask shape and how batching works

        # adj, mask, x = data.adj, data.mask, data.x
        x_all, l_total, e_total = [], 0, 0

        for i in range(self.num_diffpool_layers):
            if i != 0:
                mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])

            l_total += l
            e_total += e

        x = self.final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diffpool layers)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        if self.training:
            return (x, l_total, e_total)
        else: return x

    def loss(self, out, targets):
        preds, lp_loss, ent_loss = out
        CEloss = F.cross_entropy(preds, targets, reduction='mean')
        return CEloss + lp_loss + ent_loss


# DGCNN

class DGCNN(nn.Module):
    """
    Uses fixed architecture
    """

    def __init__(self, dim_features, dim_target, k, 
                embedding_dim=32, hidden_dense_dim=128, num_layers=3):
        super(DGCNN, self).__init__()

        self.k = k
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dim_target=dim_target

        self.convs = []
        for layer in range(self.num_layers):
            input_dim = dim_features if layer == 0 else self.embedding_dim
            self.convs.append(DGCNNConv(input_dim, self.embedding_dim))
        self.total_latent_dim = self.num_layers * self.embedding_dim

        # Add last embedding
        self.convs.append(DGCNNConv(self.embedding_dim, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)

        # should we leave this fixed?
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.input_dense_dim = (dense_dim - 5 + 1) * 32

        self.hidden_dense_dim = hidden_dense_dim
        self.dense_layer = nn.Sequential(nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(self.hidden_dense_dim, dim_target))

        #new 
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        #self.global_pool = aggr.SortAggregation(k=self.k)

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_repres = []

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = global_sort_pool(x_to_sortpool, batch, self.k) # in the code the authors sort the last channel only
        #this doesnt work.... x_1d= self.global_pool(x_to_sortpool, batch)  

        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1)
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1)

        # apply dense layer
        out_dense = self.dense_layer(conv1d_res)
        return out_dense


class DGCNNConv(MessagePassing):
    """
    Extended from tuorial on GCNs of Pytorch Geometrics
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



