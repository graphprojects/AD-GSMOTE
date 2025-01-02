import numpy as np
import torch


def get_augmentation(train_label, tro):

    label_count_dict = {}
    for i in range(train_label.max() + 1):
        label_count_dict[i] = train_label.tolist().count(i)
    label_count_dict = dict(sorted(label_count_dict.items()))
    label_freq_array = np.array(list(label_count_dict.values())) / train_label.shape[0]
    label_augment = torch.from_numpy(np.log(label_freq_array**tro + 1e-12))
    return label_augment
