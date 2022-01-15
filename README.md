# Neighbourhood Retrieval with Distance Correlation

Assign Pseudo class labels to datapoints in the latent space.

- NNDC is a slim wrapper around [FAISS](https://github.com/facebookresearch/faiss).
- NNDC transforms the space such that the Inner Product Index in FAISS (IndexFlatIP) computes the [Distance Correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html).
- Support for KernelPCA (non-linear PCA) for dimensionality reduction.

## Installation
`pip install git+https://github.com/The-Learning-Machines/nndc`

## Usage
```python
dim = 128 
n = 20000

import nndc
import numpy as np

index = nndc.DCIndex(
    in_dim=dim, 
    num_points=n, 
    threshold=0.2, 
    out_dim=32, 
    use_pca=True, 
    verbose=True,
    kernel="rbf"
)

# Generate Random data
np.random.seed(1234)             
xb = np.random.random((n, dim)).astype('float32')
xb[:, 0] += np.arange(n) / 1000.
xq = np.random.random((100, dim)).astype('float32')
xq[:, 0] += np.arange(100) / 1000.

# Fit KernelPCA
index.add_pca_training_data(xb[:1000, :])
index.fit_pca()

# Add vectors to the Index
vector_ids = np.arange(xb.shape[0])
index.add(xb, vector_ids)

# Build a nerighbourhood graph
index.build_neighbourhood()

# Query the neighbours of vector with ID=0
neighbour_ids, neighbour_similarity = index[0]   
```