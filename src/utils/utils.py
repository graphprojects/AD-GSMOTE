from sklearn.metrics import f1_score
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import random
from scipy.spatial.distance import pdist, squareform
from imblearn.metrics import geometric_mean_score
from scipy.io import loadmat
from easydict import EasyDict


def get_performance(output, labels, pre="valid"):
    gmean = geometric_mean_score(
        labels.cpu().detach(),
        torch.argmax(output, dim=-1).cpu().detach(),
        average="multiclass" if labels.max() > 1 else "binary",
        correction=0.001,
    )
    macro_f1 = f1_score(
        labels.cpu().detach(),
        torch.argmax(output, dim=-1).cpu().detach(),
        average="macro",
    )

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()

    log_dict = EasyDict(
        {
            "gmean": gmean,
            "macro_f1": macro_f1,
            "correct": correct,
        }
    )
    return log_dict


def recon_upsample_degrees_dict(
    embed,
    labels,
    idx_train,
    k,
    device,
    adj,
    im_class_num,
    portion=0.0,
    dynamic=1,
    dynamic_model=0,
):
    """Reconstruct and upsample nodes based on degree distribution for imbalanced classes.

    This function performs synthetic node generation for classes with fewer samples (tail classes)
    by considering node degrees and neighborhood information. It creates new nodes by:
    1. Identifying tail nodes (nodes with degree <= k)
    2. Generating synthetic embeddings for each tail class
    3. Creating connections between synthetic and original nodes
    4. Adding center nodes for each class

    Args:
        embed (torch.Tensor): Node feature embeddings
        labels (torch.Tensor): Node labels
        idx_train (torch.Tensor): Indices of training nodes
        k (int): Degree threshold for identifying tail nodes
        device (torch.device): Device to run computations on
        adj (torch.Tensor): Adjacency matrix
        im_class_num (int): Number of imbalanced classes
        portion (float, optional): Portion of nodes to generate. If 0, calculated automatically. Defaults to 0.0
        dynamic (int, optional): Whether to use dynamic node generation. Defaults to 1
        dynamic_model (int, optional): Dynamic model type if dynamic=1. Defaults to 0

    Returns:
        EasyDict: Dictionary containing:
            - labels: Updated node labels including synthetic nodes
            - idx_train: Updated training indices
            - new_adj: Updated adjacency matrix
            - chosen_tail_list: List of selected tail nodes
            - first_neighbor_list: First-order neighbors of tail nodes
            - second_neighbor_list: Second-order neighbors of tail nodes
            - n_tail_class: Statistics about tail nodes per class
    """
    embed = embed.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    adj = adj.to(device).to_dense()

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    adj_new = None
    chosen_tail_list, tail_new_list = [], []
    tail_idx_neighbor_list, tail_idx_neighbor_second_list = [], []
    first_neighbor_list, second_neighbor_list = [], []

    # Calculate node degrees and identify tail/head nodes
    num_egdes = np.sum(adj.cpu().numpy(), axis=1)
    tail_nodes = np.where(num_egdes <= k)[0]  # Nodes with degree <= k
    head_nodes = np.where(num_egdes > k)[0]  # Nodes with degree > k

    tail_up_count = 0
    # Handle nodes without labels by assigning -1
    labels_other = torch.tensor(
        [-1 for i in range(embed.shape[0] - labels.shape[0])]
    ).to(device)
    labels = torch.cat((labels, labels_other), dim=0)

    # Initialize center node adjacency matrix
    adj_center = torch.zeros(im_class_num, adj.shape[0]).to(device)
    class_newnode_count_dict = {}
    feature_centernode = torch.empty((im_class_num, embed.shape[1])).to(device)
    class_newnode_dict = {}
    n_tail_class = []  # Stores [num_tail_nodes, num_synthetic_nodes] for each class
    tail_correspond_syn_array_dict = {}

    # Process each imbalanced class
    for i in range(im_class_num):
        tail_correspond_syn = []
        tail_new_class_list = []
        class_newnode_count_dict[i] = 0
        tail_list, tail_idx_list = [], []

        # Get nodes belonging to current class
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        # Calculate center node embedding as mean of class embeddings
        feature_centernode[i] = torch.mean(embed[chosen, :], dim=0)

        chosen_copy = chosen.clone()
        chosen_list = chosen.tolist()
        for k in range(len(chosen_list)):
            if chosen_list[k] in tail_nodes:
                tail_list.append(chosen_list[k])
                tail_idx_list.append(k)
        chosen_tail = torch.tensor(tail_list)
        chosen_tail_idx = torch.tensor(tail_idx_list)
        # chosen_tail = torch.tensor([i for i in chosen.tolist() if i in tail_nodes])
        if chosen_tail.shape[0] != 0:
            chosen = chosen_tail
            # print("the label is {}".format(c_largest - i))
            # print("the number of tail nodes is {}".format(chosen.shape[0]))
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0] * 0.3)
        else:
            c_portion = 1
        n_tail_class.append([chosen.shape[0], c_portion])
        # print("the run number for fake node generation is {}".format(c_portion))
        for _ in range(c_portion):
            chosen_embed_class = embed[chosen_copy, :]  # embed of nodes in this class
            distance = squareform(
                pdist(chosen_embed_class.cpu().detach())
            )  # calculate the distance among the same class
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(
                axis=-1
            )  # 130 get the index of min distance among the same class
            tail_idx_neighbor = idx_neighbor[
                chosen_tail_idx
            ]  # 32 get the index of the tail nodes
            chosen_embed = embed[chosen_copy, :]  # 130 the embed of all nodes
            try:
                idx_neighbor_second = np.argsort(distance, axis=0)[
                    1, :
                ]  # 130 get the index of the second minimum distance
            except:
                idx_neighbor_second = idx_neighbor
            tail_idx_neighbor_second = idx_neighbor_second[
                chosen_tail_idx
            ]  # 32 get the index of the tail nodes

            chosen_tail_list += chosen.tolist()
            first_neighbor_index = chosen_copy[tail_idx_neighbor].reshape(-1).tolist()
            second_neighbor_index = (
                chosen_copy[tail_idx_neighbor_second].reshape(-1).tolist()
            )
            first_neighbor_list += chosen_copy[tail_idx_neighbor].reshape(-1).tolist()
            second_neighbor_list += (
                chosen_copy[tail_idx_neighbor_second].reshape(-1).tolist()
            )
            if dynamic == 0:
                interp_place_center = random.random()
                new_embed = (
                    embed[chosen, :] + feature_centernode[i] * interp_place_center
                )

            else:
                new_embed = dynamic_model(
                    embed,
                    chosen_embed,
                    chosen,
                    tail_idx_neighbor,
                    tail_idx_neighbor_second,
                    tail_up_count,
                )
                tail_idx_neighbor_list += tail_idx_neighbor.tolist()
                tail_idx_neighbor_second_list += tail_idx_neighbor_second.tolist()

            tail_up_count += len(chosen)

            new_labels = (
                labels.new(torch.Size((chosen.shape[0], 1)))
                .reshape(-1)
                .fill_(c_largest - i)
            )
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            tail_new_list += list(idx_new)
            tail_new_class_list += list(idx_new)
            idx_train_append = idx_train.new(idx_new)
            tail_correspond_syn.append(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(
                        torch.clamp_(
                            adj[chosen, :] + adj[first_neighbor_index, :],
                            min=0.0,
                            max=1.0,
                        )
                    )

                else:

                    temp = adj.new(
                        torch.clamp_(
                            adj[chosen, :] + adj[first_neighbor_index, :],
                            min=0.0,
                            max=1.0,
                        )
                    )
                    adj_new = torch.cat((adj_new, temp), 0)

        class_newnode_dict[i] = tail_new_class_list
        adj_center[i, chosen] = 1

        tail_correspond_syn_array = np.array(tail_correspond_syn).T
        tail_correspond_syn_array = np.insert(
            tail_correspond_syn_array, 0, values=chosen.tolist(), axis=1
        )

        for i in range(chosen.shape[0]):
            tail_correspond_syn_array_dict[tail_correspond_syn_array[i, 0]] = (
                tail_correspond_syn_array[i, 1:]
            )

    labels_centernode = torch.tensor([c_largest - i for i in range(im_class_num)]).to(
        device
    )
    idx_centernode = np.arange(embed.shape[0], embed.shape[0] + im_class_num)

    embed = torch.cat((embed, feature_centernode), dim=0)
    labels = torch.cat((labels, labels_centernode), dim=0)
    idx_train = torch.cat((idx_train, idx_train.new(idx_centernode)))
    # Append the adjacent matrix of newly added nodes at the end

    adj_new = torch.cat((adj_new, adj_center))

    add_num = adj_new.shape[0]
    new_adj = adj.new(
        torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))
    ).fill_(0.0)

    new_adj[: adj.shape[0], : adj.shape[0]] = adj[
        :, :
    ]  # original adjacency matrix (top-left)

    new_adj[adj.shape[0] :, : adj.shape[0]] = adj_new[
        :, :
    ]  # new adjacency matrix (bottom-left)

    for k, v in class_newnode_dict.items():
        new_adj[-(im_class_num - k), v] = (
            1  # connect new generated nodes with center nodes (bottom-right)
        )
        new_adj[v, -(im_class_num - k)] = (
            1  # connect new generated nodes with center nodes (top-right)
        )

    new_adj = new_adj.cpu().to_sparse().to(device)
    # print("the number of new nodes is {}".format(add_num))

    return EasyDict(
        {
            "labels": labels.long(),
            "idx_train": idx_train,
            "new_adj": new_adj.detach(),
            "chosen_tail_list": torch.tensor(chosen_tail_list).long(),
            "first_neighbor_list": torch.tensor(first_neighbor_list),
            "second_neighbor_list": torch.tensor(second_neighbor_list),
            "n_tail_class": n_tail_class,
            # "tail_correspond_syn_array_dict": tail_correspond_syn_array_dict,
        }
    )


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def seed_torch(seed=1029):
    """Set random seeds for reproducibility across all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # Disable hash randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def update_center_nodes(outputs, dataset, im_class_num):
    # TODO Add Docstring
    features = dataset.features
    adj_up_list = dataset.adj_up_list
    idx_train_new_list = dataset.idx_train_new_list
    label_new_list = dataset.label_new_list
    train_correct_indexs_list = dataset.train_correct_indexs_list
    center_dict_lists = dataset.center_dict_lists
    for i in range(len(adj_up_list)):
        log_dict = get_performance(
            outputs[i][idx_train_new_list[i]],
            label_new_list[i][idx_train_new_list[i]],
            pre="adj_{}".format(i),
        )
        train_correct_indexs_list[i] = torch.cat(
            (
                train_correct_indexs_list[i],
                idx_train_new_list[i][
                    torch.where(log_dict.correct == 1.0)[0].reshape(1, -1)
                ],
            ),
            dim=1,
        )

        for j in range(1, im_class_num + 1):
            center_dict_lists[i][j] = (
                train_correct_indexs_list[i][
                    train_correct_indexs_list[i] < features.shape[0]
                ]
                .reshape(-1, 1)[
                    label_new_list[i][
                        train_correct_indexs_list[i][
                            train_correct_indexs_list[i] < features.shape[0]
                        ].long()
                    ]
                    == j
                ]
                .reshape(1, -1)
            )

    dataset.center_dict_lists = center_dict_lists
    dataset.train_correct_indexs_list = train_correct_indexs_list
    return dataset
