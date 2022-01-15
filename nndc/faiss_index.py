import faiss
import numpy as np
from sklearn.decomposition import KernelPCA

class DCIndex:
    ''' 
    For assigning Pseudo-labels to datapoints in the latent space.
    Builds an In-memory Distance Correlation Index with N datapoints.
    '''
    def __init__(self, in_dim, num_points, threshold, out_dim=None, use_pca=False, kernel='rbf', n_jobs=-1, verbose=False):
        '''
        Args:
            in_dim: Dimensionality of the Input vectors
            num_points: Number of vectors in the dataset (used for building the distance matrix)
            threshold: The maximum distance of a vector from a query vector to be considered a neighbour. Note that the "Distance" is actually = 1 - Correlation Distance.
            out_dim: Dimensionality of the projected vectors in the index if using PCA.
            use_pca: Whether to use KernelPCA or not.
            kernel: Kernel to use for KernelPCA. Can be `linear`, `poly`, `rbf`, `sigmoid`, `cosine`, `precomputed`.
            n_jobs: Number of workers to use for training PCA. Default: use all available cores.
        '''
        self.use_pca = use_pca
        self.num_points = num_points
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.threshold = threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.kernel = kernel

        if use_pca:
            assert not out_dim is None, "Number of PCA components must be specified!"

        self.init()

    def init(self):
        '''Initialize PCA and FAISS Index objects'''
        if self.use_pca:
            self.pca = KernelPCA(self.out_dim, kernel=self.kernel, n_jobs=self.n_jobs)
        dset_dim = self.out_dim if not self.out_dim is None else self.in_dim
        # Inner Product Index
        self.index = faiss.IndexFlatIP(dset_dim)
        if self.use_pca:
            self.pca_train_data = []
            self.is_trained = False
        self.id_to_neighbour = {}
        self.inner_id_to_outer_id = {}
        self.outer_id_to_inner_id = {}
        self.counter = 0

    def fit_pca(self):
        '''
        Fit PCA on vectors added using `add_pca_training_data`
        '''
        data = np.concatenate(self.pca_train_data, axis=0).astype(np.float32)
        data = data - data.mean(axis=1).reshape(-1, 1)
        faiss.normalize_L2(data)

        self.pca.fit(data)
        
        self.pca_train_data = []
        self.is_trained = True
        if self.verbose:
            print("Trained PCA")

    def add_pca_training_data(self, x):
        '''
        Add vectors to train PCA
        x -> (b, inner_dim)
        or 
        x -> (inner_dim,)
        '''
        if len(x.shape) == 1:
            assert(x.shape[0] == self.in_dim)
            self.pca_train_data.append(x.reshape(1, -1))
        else:
            self.pca_train_data.append(x)

    def add(self, x, xids):
        '''
        Add vectors to FAISS Index
        x -> (b, inner_dim)
        xids -> (b,)
        '''
        assert(x.shape[1] == self.in_dim)
        x = x.astype(np.float32)
        # Center and normalize the vectors so that Inner Product computes Distance Correlation
        # Distance Correlation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html
        x = x - x.mean(axis=1).reshape(-1, 1)
        faiss.normalize_L2(x)
        
        start_id = self.counter
        end_id = self.counter + len(xids)
        for outer_iter, inner_id in enumerate(range(start_id, end_id)):
            self.inner_id_to_outer_id[inner_id] = int(xids[outer_iter])
            self.outer_id_to_inner_id[int(xids[outer_iter])] = inner_id
            
        if self.use_pca:
            assert self.is_trained, "PCA must be trained before adding vectors to the Index!"
            self.index.add(
                self.pca.transform(
                    x
                )
            )
        else:
            self.index.add(
                x
            )
        self.counter += len(xids)

    def build_neighbourhood(self):
        '''
        Builds a Dictionary[ID, (Neigoubourhood IDs, Distances)]
        '''
        vectors = self.index.reconstruct_n(0, self.num_points)
        lim, D, I = self.index.range_search(vectors, self.threshold)
        for i in range(self.num_points):
            transformed_ids = np.array([self.inner_id_to_outer_id[inner_id] for inner_id in I[lim[i]:lim[i+1]]])
            self.id_to_neighbour[self.inner_id_to_outer_id[i]] = (transformed_ids, D[lim[i]:lim[i+1]])

        del self.pca
        del self.index

    def __getitem__(self, idx):
        return self.id_to_neighbour[idx]
    
    def __len__(self):
        return len(self.id_to_neighbour)