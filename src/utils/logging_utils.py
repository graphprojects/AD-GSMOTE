import os
import logging


def setup_logger(group_name, log_dir="logs"):
    """Setup logger for training.

    Args:
        group_name (str): Name for the experiment group
        log_dir (str): Directory to store log files

    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{group_name}.log")

    logger = logging.getLogger(group_name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
