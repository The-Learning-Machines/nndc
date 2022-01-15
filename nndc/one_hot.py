import numpy as np

def one_hot(t, num_classes):
    temp = t.reshape(-1)
    res = np.zeros((len(temp), num_classes), dtype=np.int64)
    res[np.arange(len(temp)), temp] = 1
    return res.reshape(*t.shape, num_classes)