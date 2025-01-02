import sys

sys.dont_write_bytecode = True

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphSageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphSageConv, self).__init__()

        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

        self.reset_parameters()

        # print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        # fuse info from neighbors. to be added:
        if not isinstance(adj, torch.sparse.Tensor):
            # if not isinstance(adj, torch.sparse.FloatTensor):
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (
                    adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1
                )
            else:
                neigh_feature = torch.mm(adj, features) / (
                    adj.sum(dim=1).reshape(adj.shape[0], -1) + 1
                )
        else:
            # print("spmm not implemented for batch training. Note!")

            neigh_feature = torch.spmm(adj, features) / (
                adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1
            )

        # perform conv
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GraphSAGE, self).__init__()

        self.sage1 = GraphSageConv(nfeat, nhid)
        self.sage2 = GraphSageConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(F.relu(self.sage2(x, adj)))
        return x
