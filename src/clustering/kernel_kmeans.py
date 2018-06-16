import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


class KernelKMeans(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    def fit(self, K, y=None, sample_weight=None):
        n_samples = K.shape[0]

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_, update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster has not changed since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                break

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for i in range(self.n_clusters):
            mask = self.labels_ == i

            if np.sum(mask) == 0:
                # print("Empty cluster found, try smaller n_cluster")
                return False
                # raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[i] = dist_j
                dist[:, i] += dist_j
            else:
                dist[:, i] += within_distances[i]

            dist[:, i] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, K):
        n_samples = K.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_, update_within=False)
        return dist.argmin(axis=1)