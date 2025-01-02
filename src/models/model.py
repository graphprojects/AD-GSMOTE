import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GNNs import GCN, GraphSAGE

sys.dont_write_bytecode = True


class SemanticAggregate(nn.Module):
    """Semantic Aggregation Layer for Graph Neural Networks.

    This layer performs semantic-level aggregation of node embeddings using an attention mechanism.
    It learns to weight and combine node embeddings based on their semantic importance.

    Args:
        in_dim (int): Input dimension of node embeddings
        att_dim (int): Dimension of the attention space

    Attributes:
        w (nn.Parameter): Weight matrix to transform embeddings into attention space
        b (nn.Parameter): Bias vector for attention space transformation
        a (nn.Parameter): Attention vector to compute attention weights
        tanh (nn.Tanh): Tanh activation function
    """

    def __init__(self, in_dim, att_dim):
        super(SemanticAggregate, self).__init__()

        self.w = nn.Parameter(torch.empty(in_dim, att_dim))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.empty(att_dim))
        nn.init.zeros_(self.b)

        self.a = nn.Parameter(torch.empty(att_dim, 1))
        nn.init.xavier_uniform_(self.a)

        self.tanh = nn.Tanh()

    def forward(self, embed):
        trans = self.tanh(torch.add(torch.matmul(embed, self.w), self.b))
        w_r = torch.matmul(trans, self.a)

        alpha = F.softmax(w_r, dim=1)

        output = torch.sum(embed * alpha, dim=1)

        return output


class DynamicSmote(nn.Module):
    """Dynamic SMOTE module for graph data augmentation.

    This module implements a dynamic version of SMOTE (Synthetic Minority Over-sampling Technique)
    adapted for graph-structured data. It generates synthetic samples for minority classes by
    interpolating between existing samples and their class centers.

    Args:
        in_dim (int): Input dimension of node features

    Attributes:
        sm_weight_center (nn.Parameter): Learnable weight matrix for interpolation between
            samples and class centers. Shape: (in_dim, in_dim)

    The forward pass takes:
        - data: An EasyDict containing graph data including features, labels, chosen tail nodes,
               and center dictionaries
        - edge_type_index (int): Index specifying which edge type to process

    Returns:
        Synthetic node embeddings generated through interpolation between tail nodes and
        their class centers
    """

    def __init__(self, in_dim):
        super(DynamicSmote, self).__init__()

        self.sm_weight_center = nn.Parameter(torch.rand(in_dim, in_dim))
        nn.init.xavier_uniform_(self.sm_weight_center)

    def forward(
        self,
        data,
        edge_type_index,
    ):

        embed = data.features
        label = data.labels
        chosen_tails = data.chosen_tail_lists[edge_type_index]
        center_dict = data.center_dict_lists[edge_type_index]

        center_embed = []

        if center_dict == [{}, {}, {}, {}]:

            for i in range(1, label.max() + 1):
                class_index = torch.where(label == i)
                center_embed.append(
                    torch.mean(embed[class_index], axis=0).reshape(1, -1)
                )
        else:
            for i in range(1, label.max() + 1):
                if center_dict[i].shape[1] == 0:

                    class_index = torch.where(label == i)
                    center_embed.append(
                        torch.mean(embed[class_index], axis=0).reshape(1, -1)
                    )
                else:
                    center_embed.append(
                        torch.mean(
                            embed[center_dict[i].reshape(1, -1).long()].squeeze(0),
                            axis=0,
                        ).reshape(1, -1)
                    )

        center_embed_chosen_tails = torch.stack(center_embed)[
            (label[chosen_tails] - 1)
        ].squeeze(1)

        new_embed = torch.add(
            embed[chosen_tails, :],
            torch.sub(embed[chosen_tails, :], center_embed_chosen_tails).matmul(
                self.sm_weight_center
            ),
        )

        return torch.cat(
            (embed[: label.shape[0]], new_embed, torch.stack(center_embed).squeeze(1)),
            dim=0,
        )


class DynSMOTE_Encoder(nn.Module):
    """Dynamic SMOTE Encoder module that combines dynamic SMOTE sampling with GNN backbone.

    This module performs dynamic synthetic minority oversampling (SMOTE) followed by graph neural
    network encoding. It supports GCN and GraphSAGE as backbone architectures.

    Args:
        model_config: Configuration object containing:
            - in_dim (int): Input feature dimension
            - hid_dim (int): Hidden layer dimension
            - dropout (float): Dropout probability
            - backbone (str): GNN backbone architecture ('GCN' or 'SAGE')

    Attributes:
        dyn_smote (DynamicSmote): Dynamic SMOTE sampling module
        GNN (nn.Module): Graph neural network backbone (GCN or GraphSAGE)
    """

    def __init__(self, model_config):
        super(DynSMOTE_Encoder, self).__init__()

        in_dim = model_config.in_dim
        hid_dim = model_config.hid_dim
        dropout = model_config.dropout

        self.dyn_smote = DynamicSmote(in_dim)
        if model_config.backbone == "GCN":
            self.GNN = GCN(nfeat=in_dim, nhid=hid_dim, dropout=dropout)
        elif model_config.backbone == "SAGE":
            self.GNN = GraphSAGE(nfeat=in_dim, nhid=hid_dim, dropout=dropout)
        else:
            raise ValueError(f"Backbone {model_config.backbone} not supported")

    def forward(self, data):

        adj_up_list = data.adj_up_list

        embed_ds = []
        for i in range(len(adj_up_list)):
            features_ds = self.dyn_smote(data, edge_type_index=i)
            embed_ds.append(self.GNN(features_ds, adj_up_list[i]))

        return embed_ds


class AD_GSMOTE(nn.Module):
    """Adaptive Dynamic Graph SMOTE model for imbalanced node classification.

    This model combines dynamic SMOTE sampling with semantic aggregation of multiple graph views
    to address class imbalance and topological imbalance in graph neural networks. It consists of:
    1. A DynSMOTE encoder that performs dynamic synthetic minority oversampling
    2. A semantic aggregator that combines embeddings from multiple graph views
    3. A classification head for the downstream task

    Args:
        model_config: Configuration object containing model hyperparameters:
            - num_nodes (int): Number of nodes in the graph
            - hid_dim (int): Hidden layer dimension
            - att_dim (int): Attention dimension for semantic aggregation
            - num_classes (int): Number of output classes
            - in_dim (int): Input feature dimension
            - dropout (float): Dropout probability
            - backbone (str): GNN backbone architecture ('GCN' or 'SAGE')

    Attributes:
        encoder (DynSMOTE_Encoder): Dynamic SMOTE encoder module
        fc (nn.Linear): Classification head
        semanticaggregator (SemanticAggregate): Module to aggregate multiple views
        criterion (nn.CrossEntropyLoss): Loss function for training
    """

    def __init__(self, model_config):
        super(AD_GSMOTE, self).__init__()

        self.model_config = model_config

        self.num_nodes = model_config.num_nodes

        hid_dim = model_config.hid_dim
        att_dim = model_config.att_dim
        num_classes = model_config.num_classes

        self.encoder = DynSMOTE_Encoder(model_config)

        self.fc = nn.Linear(hid_dim, num_classes)
        self.semanticaggregator = SemanticAggregate(in_dim=hid_dim, att_dim=att_dim)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        embed_ds_list = self.encoder(data)
        embeds = embed_ds_list[0][: self.num_nodes].reshape(self.num_nodes, 1, -1)
        for i in range(1, len(embed_ds_list)):
            embeds = torch.cat(
                (
                    embeds,
                    embed_ds_list[i][: self.num_nodes].reshape(self.num_nodes, 1, -1),
                ),
                dim=1,
            )
        output = self.semanticaggregator(embeds)

        return embed_ds_list, output

    def loss(self, data, mode="train"):

        labels = data.labels
        idx = data[f"idx_{mode}"]
        logits_augment = data.logits_augment
        idx_train_new_list = data.idx_train_new_list
        label_new_list = data.label_new_list

        loss = []
        output_list = []

        embeds_ds_list, embeds = self.forward(data)

        # TODO Ablation
        output_task = self.fc(embeds) + logits_augment

        for i in range(len(embeds_ds_list)):

            # TODO Ablation
            outputs = self.fc(embeds_ds_list[i]) + logits_augment
            output_list.append(outputs)
            loss.append(
                self.criterion(
                    outputs[idx_train_new_list[i]],
                    label_new_list[i][idx_train_new_list[i]],
                )
            )

        output_list.append(output_task)
        loss_task = self.criterion(output_task[idx], labels[idx])

        loss_final = loss_task + torch.sum(torch.stack(loss))

        return loss_final, output_list
