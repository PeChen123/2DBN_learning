import numpy as np
import networkx as nx
import igraph as ig

import numpy as np
import igraph as ig

class DataSim:
    def __init__(self, sample_size, d_size, degree, eta, lag):
        self.size = sample_size
        self.d = d_size
        self.p = degree  # This is used as an edge probability
        self.eta = eta
        self.lag = lag

    def simulate_dag(self):
        """Simulate random DAG with some expected number of edges.

        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """
        def _random_permutation(M):
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        # Generate an undirected Erdős-Rényi graph with probability p
        G_und = ig.Graph.Erdos_Renyi(n=self.d, p=self.p/self.d)
        B_und = _graph_to_adjmat(G_und)
        # Ensure the graph is acyclic by using a lower triangular matrix
        B = _random_acyclic_orientation(B_und)
        # Randomly permute the nodes to ensure the DAG is not trivially ordered
        B_perm = _random_permutation(B)

        # Check if the graph is a DAG
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm

    def is_dag(self, W):
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def W_sim(self):
        """Simulate weight matrix W based on the DAG structure."""
        # Generate the binary adjacency matrix for a DAG
        B = self.simulate_dag()

        # Assign weights to the edges in the DAG
        W = B.astype(float)
        for i in range(self.d):
            for j in range(self.d):
                if W[i, j] == 1:
                    # Assign weights from a uniform distribution over the specified range
                    W[i, j] = np.random.uniform(-2.0, -0.5) if np.random.rand() < 0.5 else np.random.uniform(0.5, 2.0)

        # Apply threshold to eliminate very small values
        threshold = 0.01
        W[np.abs(W) < threshold] = 0

        return W

    def A_sim(self):
        """Simulate lagged inter-slice influence matrices A_1, A_2, ..., A_p."""
        A_weighted_matrices = []

        # Generate pairwise inter-slice influence matrices for each lag
        for l in range(1, self.lag + 1):
            matrix_size_pairwise = 2 * self.d  # 10 x 10 matrix if d = 5
            pr_A = 1 / self.d  # Probability for each edge

            # Generate ER graph for the 2*d x 2*d matrix
            ER = ig.Graph.Erdos_Renyi(n=matrix_size_pairwise, p=pr_A, directed=True)
            A_full = np.array(ER.get_adjacency().data)

            # Extract the inter-slice part: rows [0:d], columns [d:2*d]
            A_inter_slice = A_full[0:self.d, self.d:2*self.d]

            # Assign weights to the inter-slice matrix
            A_l_weighted = np.zeros((self.d, self.d))
            alpha = 1 / (self.eta ** (l - 1))  # Decay factor for each lag

            for j in range(self.d):
                for k in range(self.d):
                    if A_inter_slice[j, k] == 1:
                        # Assign weight with decaying factor
                        A_l_weighted[j, k] = np.random.uniform(-2.0 * alpha, -0.5 * alpha) if np.random.rand() < 0.5 else np.random.uniform(0.5 * alpha, 2.0 * alpha)

            # Apply threshold to eliminate very small values
            threshold = 0.01
            A_l_weighted[np.abs(A_l_weighted) < threshold] = 0

            # Append weighted A_l matrix to the list
            A_weighted_matrices.append(A_l_weighted)

        return np.vstack(A_weighted_matrices)