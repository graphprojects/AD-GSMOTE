import os.path
import sys
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
from easydict import EasyDict

sys.dont_write_bytecode = True


def get_performance(output, labels):
    y_true = labels.cpu().detach()
    y_pred = torch.argmax(output, dim=-1).cpu().detach()

    average = "multiclass" if labels.max() > 1 else "binary"

    gmean = geometric_mean_score(y_true, y_pred, average=average, correction=0.001)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return EasyDict({"gmean": gmean, "macro_f1": macro_f1, "correct": correct})
