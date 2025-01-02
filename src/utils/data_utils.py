import sys
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.distance import pdist, squareform
import random
from easydict import EasyDict
from utils.eval_metrics import get_performance

sys.dont_write_bytecode = True


def update_center_nodes(outputs, dataset, im_class_num):
    """
    Update the center nodes for each class
    """
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
    return torch.sparse_coo_tensor(indices, values, shape)


def recon_upsample_degrees_dict(
    dataset,
    device,
    edge_type_index,
    im_class_num,
    portion=0.0,
):
    """Reconstruct upsampled degrees dictionary for imbalanced classes.

    Args:
        dataset: Dataset object containing graph data
        device: Device to run computations on (CPU/GPU)
        edge_type_index: Index of edge type to process
        im_class_num: Number of imbalanced classes
        portion: Portion of nodes to upsample (default: 0.0)

    Returns:
        Dataset object with updated attributes:
        - adj_new: Reconstructed adjacency matrix
        - chosen_tail_list: List of chosen tail nodes
        - tail_new_list: List of new tail nodes
        - first_neighbor_list: List of first-hop neighbors
        - second_neighbor_list: List of second-hop neighbors
    """

    embed = dataset.features.to(device)
    labels = dataset.labels.to(device)
    idx_train = dataset.idx_train.to(device)
    k = dataset.degree_list[edge_type_index]
    adj = dataset.adjs[edge_type_index].to(device).to_dense()

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new = None
    chosen_tail_list, tail_new_list = [], []
    first_neighbor_list, second_neighbor_list = [], []
    num_egdes = np.sum(adj.cpu().numpy(), axis=1)
    tail_nodes = np.where(num_egdes <= k)[0]
    tail_up_count = 0
    labels_other = torch.tensor(
        [-1 for _ in range(embed.shape[0] - labels.shape[0])], device=labels.device
    )
    labels = torch.cat((labels, labels_other), dim=0)
    adj_center = torch.zeros(im_class_num, adj.shape[0], device=device)
    class_newnode_count_dict = {}
    feature_centernode = torch.empty((im_class_num, embed.shape[1]), device=device)
    class_newnode_dict = {}

    for i in range(im_class_num):
        tail_correspond_syn = []
        tail_new_class_list = []
        class_newnode_count_dict[i] = 0
        tail_list, tail_idx_list = [], []
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        feature_centernode[i] = torch.mean(
            embed[chosen, :], dim=0
        )  # generate the embedding of center nodes

        chosen_copy = chosen.clone()
        chosen_list = chosen.tolist()
        for k in range(len(chosen_list)):
            if chosen_list[k] in tail_nodes:
                tail_list.append(chosen_list[k])
                tail_idx_list.append(k)
        chosen_tail = torch.tensor(tail_list)
        chosen_tail_idx = torch.tensor(tail_idx_list)
        if chosen_tail.shape[0] != 0:
            chosen = chosen_tail
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0] * 0.3)
        else:
            c_portion = 1
        for _ in range(c_portion):
            chosen_embed_class = embed[chosen_copy, :]  # embed of nodes in this class
            distance = squareform(
                pdist(chosen_embed_class.cpu().detach())
            )  # calculate the distance among the same class
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            tail_idx_neighbor = idx_neighbor[chosen_tail_idx]
            try:
                idx_neighbor_second = np.argsort(distance, axis=0)[1, :]
            except:
                idx_neighbor_second = idx_neighbor
            tail_idx_neighbor_second = idx_neighbor_second[chosen_tail_idx]
            chosen_tail_list += chosen.tolist()
            first_neighbor_index = chosen_copy[tail_idx_neighbor].reshape(-1).tolist()
            first_neighbor_list += chosen_copy[tail_idx_neighbor].reshape(-1).tolist()
            second_neighbor_list += (
                chosen_copy[tail_idx_neighbor_second].reshape(-1).tolist()
            )

            interp_place_center = random.random()
            new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center

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

    labels_centernode = torch.tensor(
        [c_largest - i for i in range(im_class_num)], device=labels.device
    )
    idx_centernode = np.arange(embed.shape[0], embed.shape[0] + im_class_num)

    embed = torch.cat((embed, feature_centernode), dim=0)
    labels = torch.cat((labels, labels_centernode), dim=0)
    idx_train = torch.cat((idx_train, idx_train.new(idx_centernode)))

    # TODO ablation. Keep old adj.
    adj_new = torch.cat((adj_new, adj_center))

    add_num = adj_new.shape[0]
    new_adj = adj.new(
        torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))
    ).fill_(0.0)

    new_adj[: adj.shape[0], : adj.shape[0]] = adj[:, :]
    new_adj[adj.shape[0] :, : adj.shape[0]] = adj_new[:, :]

    for k, v in class_newnode_dict.items():
        new_adj[-(im_class_num - k), v] = 1
        new_adj[v, -(im_class_num - k)] = 1

    new_adj = new_adj.cpu().to_sparse().to(device)

    return EasyDict(
        labels=labels.long(),
        idx_train=idx_train,
        new_adj=new_adj.detach(),  # a
        chosen_tail_list=torch.tensor(chosen_tail_list).long().to(device),
    )


def preprocess_dataset(dataset, args):
    device = args.device
    up_scale = args.up_scale
    im_class_num = args.im_class_num
    degree_list = args.degree_list
    dataset["degree_list"] = degree_list

    center_dict_lists = [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]
    adj_up_list = []
    label_new_list = []
    idx_train_new_list = []
    chosen_tail_lists = []
    for i in range(len(dataset.adjs)):

        recon_data = recon_upsample_degrees_dict(
            dataset,
            edge_type_index=i,
            device=device,
            portion=up_scale,
            im_class_num=im_class_num,
        )

        label_new_list.append(recon_data.labels)
        idx_train_new_list.append(recon_data.idx_train)
        adj_up_list.append(recon_data.new_adj)

        chosen_tail_lists.append(recon_data.chosen_tail_list)

    dataset.update(
        label_new_list=label_new_list,
        idx_train_new_list=idx_train_new_list,
        adj_up_list=adj_up_list,
        chosen_tail_lists=chosen_tail_lists,
        center_dict_lists=center_dict_lists,
    )

    dataset.idx_train = dataset.idx_train.to(device)
    dataset.idx_val = dataset.idx_val.to(device)
    dataset.idx_test = dataset.idx_test.to(device)
    dataset.labels = dataset.labels.to(device)
    dataset.adjs = [adj.to(device) for adj in dataset.adjs]
    dataset.features = dataset.features.to(device)
    dataset.train_correct_indexs_list = [
        torch.empty((1, 0), device=device),
        torch.empty((1, 0), device=device),
        torch.empty((1, 0), device=device),
        torch.empty((1, 0), device=device),
    ]
    return dataset
