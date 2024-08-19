from __future__ import absolute_import, division
from timm.models.layers import DropPath
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn.dense.linear import Linear
from src.modeling._gcnn import GraphLinear

class HE_GCN_block(nn.Module):
    def __init__(self, adj, input_dim, hidden_dim, output_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.lin1 = GraphLinear(input_dim, hidden_dim)
        self.norm_gcn1 = norm_layer(hidden_dim)
        self.gcn = ModulatedGraphConv(hidden_dim, hidden_dim, self.adj)
        self.lin2 = GraphLinear(hidden_dim, output_dim)
        self.norm_gcn2 = norm_layer(output_dim)
        self.gelu = nn.GELU()

    def forward(self, x_gcn):

        x_gcn = self.lin1(x_gcn)
        x_gcn = self.gelu(self.norm_gcn1(x_gcn))
        x_gcn = self.gcn(x_gcn)
        x_gcn = self.lin2(x_gcn)
        x_gcn = self.gelu(self.norm_gcn2(x_gcn))

        return x_gcn
class GCN_block(nn.Module):
    def __init__(self, adj, input_dim, hidden_dim, output_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.norm_gcn1 = norm_layer(input_dim)
        self.gcn1 = ModulatedGraphConv(input_dim,hidden_dim,self.adj)
        self.gelu = nn.GELU()
        self.gcn2 = ModulatedGraphConv(hidden_dim, output_dim, self.adj)
        self.norm_gcn2 = norm_layer(output_dim)

    def forward(self, x_gcn):
        # x_gcn = x_gcn + self.drop_path(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))
        x_gcn = x_gcn + self.drop_path(self.norm_gcn2(self.gcn2(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))))
        return x_gcn

class ModulatedGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))  #torch.Size([2,2, 384])
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))#17,384,取值在
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj = adj

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])  #input 256,17,2  -> 256,17,384
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj)/2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device) #17*17的I

        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1) #前者是专门针对自己的I，后者是针对M的
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1) #torch.Size([256, 17, 384])，全部都有的
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx) #+ sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])
    return adj_mx


def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)