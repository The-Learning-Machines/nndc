import torch
import numpy as np

def one_hot(t, num_classes, numpy=True):
    if not numpy:
        temp = t.reshape(-1)
        res = torch.zeros((len(temp), num_classes), dtype=torch.long)
        res[torch.arange(len(temp)), temp] = 1
        return res.reshape(*t.shape, num_classes)
    else:
        temp = t.reshape(-1)
        res = np.zeros((len(temp), num_classes), dtype=np.int64)
        res[np.arange(len(temp)), temp] = 1
        return res.reshape(*t.shape, num_classes)