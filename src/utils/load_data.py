import sys
import os.path as osp
import random

import numpy as np
import torch
from scipy.io import loadmat, savemat
from easydict import EasyDict
import scipy.sparse as sp

sys.dont_write_bytecode = True

from utils.data_utils import normalize, sparse_mx_to_torch_sparse_tensor


def load_amazon_data(
    data_dir,
    dataname="amazon",
    train_ratio=0.4,
):
    """Load amazon network dataset"""

    amazon = loadmat(osp.join(data_dir, dataname, "Amazon.mat"))

    features = normalize(amazon["features"])
    label = amazon["label"].ravel()
    adj = amazon["homo"]
    adj1 = amazon["net_upu"]
    adj2 = amazon["net_usu"]
    adj3 = amazon["net_uvu"]
    index = [i for i in range(label.shape[0])]
    random.shuffle(index)

    num_classes = len(set(label.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (label == i).nonzero()[0].tolist()
        c_num = len(c_idx)
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        c_num_mat[i, 0] = int(c_num * train_ratio)
        c_num_mat[i, 1] = int(c_num * 0.2)
        c_num_mat[i, 2] = int(c_num * (1 - train_ratio - 0.2))

        train_idx = train_idx + c_idx[: c_num_mat[i, 0]]

        val_idx = val_idx + c_idx[c_num_mat[i, 0] : c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = (
            test_idx
            + c_idx[
                c_num_mat[i, 0]
                + c_num_mat[i, 1] : c_num_mat[i, 0]
                + c_num_mat[i, 1]
                + c_num_mat[i, 2]
            ]
        )

    random.shuffle(train_idx)

    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(val_idx)
    idx_test = torch.LongTensor(test_idx)
    features = torch.FloatTensor(np.array(features.todense()))

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    adj3 = sparse_mx_to_torch_sparse_tensor(adj3)

    label = torch.LongTensor(label)

    dataset = EasyDict(
        adjs=[adj1, adj2, adj3],
        features=features,
        labels=label,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    return dataset


def load_twitter_data(data_dir, dataname="twitter_drug", train_ratio=0.4):
    """Load Twitter network dataset"""

    twitter = loadmat(osp.join(data_dir, dataname, "twitter_drug.mat"))

    features = twitter["features"]

    label = twitter["label"].ravel()
    adj1 = twitter["net_uku"]
    adj2 = twitter["net_ufu"]
    adj3 = twitter["net_utu"]

    index = [i for i in range(label.shape[0])]
    random.shuffle(index)

    num_classes = len(set(label.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (label == i).nonzero()[0].tolist()
        c_num = len(c_idx)
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        c_num_mat[i, 0] = int(c_num * train_ratio)
        c_num_mat[i, 1] = int(c_num * 0.2)
        c_num_mat[i, 2] = int(c_num * (1 - train_ratio - 0.2))

        train_idx = train_idx + c_idx[: c_num_mat[i, 0]]

        val_idx = val_idx + c_idx[c_num_mat[i, 0] : c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = (
            test_idx
            + c_idx[
                c_num_mat[i, 0]
                + c_num_mat[i, 1] : c_num_mat[i, 0]
                + c_num_mat[i, 1]
                + c_num_mat[i, 2]
            ]
        )

    random.shuffle(train_idx)

    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(val_idx)
    idx_test = torch.LongTensor(test_idx)

    features = torch.FloatTensor(np.array(features.todense()))
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    adj3 = sparse_mx_to_torch_sparse_tensor(adj3)

    label = torch.LongTensor(label)

    dataset = EasyDict(
        adjs=[adj1, adj2, adj3],
        features=features,
        labels=label,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    return dataset


def load_yelp_data(data_dir, dataname="yelpchi", train_ratio=0.4):
    """Load yelp network dataset"""

    yelp = loadmat(osp.join(data_dir, dataname, "YelpChi.mat"))

    features = normalize(yelp["features"])

    label = yelp["label"].ravel()

    adj = yelp["homo"]

    adj1 = yelp["net_rur"]

    adj2 = yelp["net_rtr"]

    adj3 = yelp["net_rsr"]

    index = [i for i in range(label.shape[0])]
    random.shuffle(index)

    num_classes = len(set(label.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (label == i).nonzero()[0].tolist()
        c_num = len(c_idx)
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        c_num_mat[i, 0] = int(c_num * train_ratio)
        c_num_mat[i, 1] = int(c_num * 0.2)
        c_num_mat[i, 2] = int(c_num * (1 - train_ratio - 0.2))

        train_idx = train_idx + c_idx[: c_num_mat[i, 0]]

        val_idx = val_idx + c_idx[c_num_mat[i, 0] : c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = (
            test_idx
            + c_idx[
                c_num_mat[i, 0]
                + c_num_mat[i, 1] : c_num_mat[i, 0]
                + c_num_mat[i, 1]
                + c_num_mat[i, 2]
            ]
        )

    random.shuffle(train_idx)

    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(val_idx)
    idx_test = torch.LongTensor(test_idx)

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    adj3 = sparse_mx_to_torch_sparse_tensor(adj3)

    label = torch.LongTensor(label)

    return EasyDict(
        adjs=[adj1, adj2, adj3],
        features=features,
        labels=label,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def load_dataset(args):
    """
    Load the dataset
    """
    train_ratio = args.train_ratio
    dataset_name = args.dataset

    if dataset_name == "twitter_drug":
        dataset = load_twitter_data(args.data_dir, train_ratio=train_ratio)
    elif dataset_name == "yelpchi":
        dataset = load_yelp_data(args.data_dir, train_ratio=train_ratio)
    elif dataset_name == "amazon":
        dataset = load_amazon_data(args.data_dir, train_ratio=train_ratio)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset
