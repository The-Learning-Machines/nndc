import sys
sys.path.append(".")

dim = 128 
n = 20000

import nndc
import numpy as np

def test_dc():
    index = nndc.DCIndex(in_dim=dim, threshold=0.2, out_dim=32, use_pca=True, verbose=True)

    np.random.seed(1234)             
    xb = np.random.random((n, dim)).astype('float32')
    xb[:, 0] += np.arange(n) / 1000.
    xq = np.random.random((100, dim)).astype('float32')
    xq[:, 0] += np.arange(100) / 1000.

    index.add_pca_training_data(xb[:1000, :])
    index.fit_pca()

    index.add(xb, np.arange(xb.shape[0]))
    index.build_neighbourhood()
    print(index[0])   