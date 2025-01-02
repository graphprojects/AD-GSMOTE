"""
Command line argument parsing for AD-GSMOTE.

This module handles the configuration of command line arguments for the AD-GSMOTE model.
"""

import os
import os.path as osp
import argparse
import torch
import ast


def parse_args(args_list=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AD-GSMOTE Training")

    # Get directory of current file for default data path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--root_dir",
        type=str,
        default=current_dir,
        help="Root directory containing the data files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=osp.join(current_dir, "data"),
        help="Data directory containing the data files",
    )
    # Basic settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="twitter_drug",
        choices=["twitter_drug", "yelpchi", "amazon"],
        help="Dataset to use",
    )
    parser.add_argument("--use_cuda", default=True, help="Disables CUDA training")
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    parser.add_argument("--print_interval", type=int, default=-1, help="Print interval")
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disables WandB logging",
    )

    # Training parameters
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to train")
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay (L2 loss)"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.4,
        help="Fraction of data to use for training, we then use rest 20% for validation and others for testing",
    )

    # Model parameters
    parser.add_argument(
        "--load_best_params",
        action="store_true",
        help="Loads the best parameters from the dataset",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=None, help="Number of hidden units"
    )
    parser.add_argument("--att_dim", type=int, default=None, help="Attention dimension")
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["GCN", "SAGE"],
        help="Type of GNN model",
    )

    # SMOTE parameters
    parser.add_argument(
        "--tro", type=float, default=None, help="Adjustment index in logit adjustment"
    )
    parser.add_argument(
        "--im_class_num", type=int, default=None, help="Number of minority classes"
    )
    parser.add_argument("--up_scale", type=float, default=0, help="Upsampling scale")
    parser.add_argument(
        "--rec_weight", type=float, default=0.000001, help="Reconstruction weight"
    )
    parser.add_argument("--degree_list", type=str, default=None, help="Degree list")
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()
    args.device = torch.device(args.device)

    if isinstance(args.degree_list, str):
        args.degree_list = ast.literal_eval(args.degree_list)

    return args
