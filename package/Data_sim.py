import numpy as np
import igraph as ig

class LinearSEMSimulator:
    def __init__(self, W, noise_scale=None):
        """
        Args:
            W (np.ndarray): [d, d] weighted adjacency matrix of DAG
            noise_scale (np.ndarray or float): scale parameter of additive noise, default all ones
        """
        if not self._is_dag(W):
            raise ValueError('W must be a DAG')
        
        self.W = W
        self.d = W.shape[0]
        self.noise_scale = noise_scale if noise_scale is not None else np.ones(self.d)
        
        if np.isscalar(self.noise_scale):
            self.noise_scale = self.noise_scale * np.ones(self.d)
        elif len(self.noise_scale) != self.d:
            raise ValueError('noise scale must be a scalar or have length d')
        
        self.G = ig.Graph.Weighted_Adjacency(self.W.tolist())
        self.ordered_vertices = self.G.topological_sorting()
        assert len(self.ordered_vertices) == self.d

    def _is_dag(self, W):
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def simulate(self, n, sem_type):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            n (int): number of samples, n=inf mimics population risk
            sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson

        Returns:
            X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
        """
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true covariance
                X = np.sqrt(self.d) * np.diag(self.noise_scale) @ np.linalg.inv(np.eye(self.d) - self.W)
                return X
            else:
                raise ValueError('population risk not available for this SEM type')
        
        X = np.zeros([n, self.d])
        for j in self.ordered_vertices:
            parents = self.G.neighbors(j, mode=ig.IN)
            X[:, j] = self._simulate_single_equation(X[:, parents], self.W[parents, j], self.noise_scale[j], n, sem_type)
        return X

    def _simulate_single_equation(self, X, w, scale, n, sem_type):
        """Simulate a single equation.

        Args:
            X (np.ndarray): [n, num of parents], input data
            w (np.ndarray): [num of parents], weights
            scale (float): noise scale
            n (int): number of samples
            sem_type (str): type of noise distribution

        Returns:
            x (np.ndarray): [n], output data
        """
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, self._sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    def create_lagged_version(self, X, lag):
        """
        Create a time-lagged version of the matrix X based on the number of lagged blocks.

        Args:
            X (np.ndarray): Original time-series data matrix of shape (n, d).
            lag (int): Number of lags.

        Returns:
            Y (np.ndarray): Lagged version of X with multiple lags.
        """
        n, d = X.shape
        Y = np.zeros((n, lag * d))

        # Fill in the lagged values for each lag
        for l in range(1, lag + 1):
            Y[l:, (l - 1) * d:l * d] = X[:-l, :]  # Assign shifted X values to the correct block in Y

        return Y

    def add_inter_slice_effects(self, X, A):
        """
        Add inter-slice (lagged) contributions from A to the generated X matrix.

        Args:
            X (np.ndarray): Generated time-series data matrix of shape (n, d).
            A (np.ndarray): Inter-slice matrix of shape (pd, d), where p is the number of lags.

        Returns:
            X_up (np.ndarray): Updated time-series data with lagged effects added.
        """
        n, d = X.shape
        lag = A.shape[0] // d  # Calculate the number of lagged blocks

        Y = self.create_lagged_version(X, lag)  # Create lagged version of X
        X_up = X + Y @ A

        return X_up

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
