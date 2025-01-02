import os
import sys
import torch
import torch.optim as optim
import time
import logging
from datetime import datetime
import numpy as np
from easydict import EasyDict

sys.dont_write_bytecode = True

from arguments import parse_args
from utils.load_params import load_dataset_params
from utils.seeds import seed_torch
from utils.load_data import load_dataset
from utils.data_utils import preprocess_dataset, update_center_nodes
from utils.eval_metrics import get_performance
from utils.logging_utils import setup_logger

from models.logit_adjustment import get_augmentation
from models import AD_GSMOTE

seed_list = [5, 10, 20, 25, 30]


def run_epoch(model, dataset, optimizer=None, mode="train"):
    """Run a single epoch of training, validation, or testing.

    Args:
        model: The neural network model
        dataset: Dataset object containing features, labels and split indices
        optimizer: PyTorch optimizer (only used in training mode)
        mode (str): One of "train", "val", or "test"

    Returns:
        dict: Dictionary containing:
            - Performance metrics (macro F1, GMean)
            - Loss value for the epoch
            - Model outputs
    """
    labels = dataset.labels

    assert mode in ["train", "val", "test"], "Invalid mode"

    if mode == "train":
        model.train()
        optimizer.zero_grad()
        idx = dataset[f"idx_{mode}"]

    else:
        model.eval()
        idx = torch.cat(
            [dataset[f"idx_train"], dataset["idx_val"], dataset["idx_test"]]
        )

    loss, outputs = model.loss(dataset)

    train_result = get_performance(outputs[-1][idx], labels[idx])

    if mode == "train":
        loss.backward()
        optimizer.step()

    train_result.update({f"{mode}_loss": loss, "outputs": outputs})

    return train_result


def main(args):
    """Main training loop that runs multiple experiments.

    Performs multiple training runs with different random seeds and reports
    averaged results. For each run:
    - Initializes model, dataset and optimizer
    - Trains for specified number of epochs
    - Tracks best validation performance
    - Saves best model weights
    - Reports final test performance

    Args:
        args: Namespace object containing all runtime arguments
    """
    macro_f1_list = []
    gmean_list = []

    start_time = time.time()
    timestamp = datetime.fromtimestamp(int(start_time)).strftime("%Y%m%d%H%M%S")
    group_name = f"{args.dataset}_{args.train_ratio:.2f}_{timestamp}"

    logger = setup_logger(group_name)
    logger.info(f"Starting experiment: {group_name}")
    logger.info(f"Arguments: {args}")

    for run in range(args.runs):
        logger.info(f"\nStarting run {run+1}/{args.runs}")

        best_val_f1, best_epoch, best_val_gmean = 0, 0, 0
        best_test_f1, best_test_gmean = 0, 0

        seed_torch(seed_list[run])
        save_model_dir = os.path.join(
            args.root_dir,
            "saved_models",
            group_name,
        )
        os.makedirs(save_model_dir, exist_ok=True)
        save_model_path = os.path.join(
            save_model_dir,
            f"{args.dataset}_{args.train_ratio}_run{run}.pkl",
        )
        dataset = load_dataset(args)
        dataset = preprocess_dataset(dataset, args)
        logits_augment = get_augmentation(dataset.labels, tro=args.tro).to(args.device)
        dataset.logits_augment = logits_augment

        model_config = EasyDict(
            in_dim=dataset.features.shape[1],
            hid_dim=args.hidden_dim,
            num_classes=dataset.labels.max() + 1,
            att_dim=args.att_dim,
            dropout=args.dropout,
            num_nodes=dataset.features.shape[0],
            backbone=args.backbone,
        )

        model = AD_GSMOTE(model_config).to(args.device)

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for epoch in range(args.epochs):
            train_result = run_epoch(model, dataset, optimizer, mode="train")
            dataset = update_center_nodes(
                train_result.outputs, dataset, args.im_class_num
            )
            val_result = run_epoch(model, dataset, mode="val")
            test_result = run_epoch(model, dataset, mode="test")

            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_result.train_loss,
                "val_loss": val_result.val_loss,
                "train_f1": train_result.macro_f1,
                "val_f1": val_result.macro_f1,
                "train_gmean": train_result.gmean,
                "val_gmean": val_result.gmean,
                "test_f1": test_result.macro_f1,
                "test_gmean": test_result.gmean,
            }

            if val_result.macro_f1 > best_val_f1:
                best_val_f1 = val_result.macro_f1
                best_test_f1 = test_result.macro_f1
                best_test_gmean = test_result.gmean
                torch.save(model.state_dict(), save_model_path)

            if (
                args.print_interval > 0
                and epoch % args.print_interval == 0
                and epoch > 0
            ):
                logger.info(
                    f"Epoch: {metrics['epoch']:04d}, "
                    f"Train Loss: {metrics['train_loss']:.2f}, "
                    f"Val Loss: {metrics['val_loss']:.2f}, "
                    f"Train F1: {metrics['train_f1']*100:.2f}%, "
                    f"Val F1: {metrics['val_f1']*100:.2f}%, "
                    f"Train GMean: {metrics['train_gmean']*100:.2f}%, "
                    f"Val GMean: {metrics['val_gmean']*100:.2f}%, "
                    f"Test F1: {metrics['test_f1']*100:.2f}%, "
                    f"Test GMean: {metrics['test_gmean']*100:.2f}%"
                )

        logger.info(
            f"Run {run+1} completed - "
            f"Best Test F1: {best_test_f1*100:.2f}%, "
            f"Best Test GMean: {best_test_gmean*100:.2f}%"
        )

        macro_f1_list.append(best_test_f1)
        gmean_list.append(best_test_gmean)

    avg_macro_f1, std_macro_f1 = np.mean(macro_f1_list), np.std(macro_f1_list)
    avg_gmean, std_gmean = np.mean(gmean_list), np.std(gmean_list)

    logger.info("\nFinal Results:")
    logger.info(f"Average Macro F1: {avg_macro_f1*100:.2f}% ± {std_macro_f1*100:.2f}%")
    logger.info(f"Average GMean: {avg_gmean*100:.2f}% ± {std_gmean*100:.2f}%")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")


def run(args_list=[]):
    """Entry point function that parses arguments and starts training.

    Args:
        args_list (list): List of command line arguments
    """
    args = parse_args(args_list)

    if args.load_best_params:
        params = load_dataset_params(args.dataset, args.train_ratio, args.root_dir)
        for key, val in params.items():
            if getattr(args, key) is None:
                setattr(args, key, val)
    main(args)


if __name__ == "__main__":
    run()
