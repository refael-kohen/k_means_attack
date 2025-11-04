# This is a toy implementation of Lloyd's algorithm for k-means with differential privacy.
# We assume that data points come from [0,1]. # REF
# This code purpose is didactic only.

import numpy as np
import matplotlib.pyplot as plt


class KMeansHeuristic:
    def __init__(self, rng, C_init, n, k, max_iter=10):
        self.max_iter = max_iter
        self._X = np.sort(np.random.choice(rng, size=n, replace=True))
        self.C_init = C_init
        self.k = k  # number of clusters
        self.n = n  # number of data points
        self._n1 = None  # number of points in cluster 1
        self._n2 = None  # number of points in cluster 2
        self.c1 = None  # center of cluster 1
        self.c2 = None  # center of cluster 2
        self._P = None  # points moved to cluster 2 after deletions
        self._n_p = None  # number of points moved to cluster 2 after deletions
        self._s_p = None # sum of points moved to cluster 2 after deletions
        # self._c2_start = None  # index of the first point classified to cluster 2

    #############################################
    # classify
    #############################################
    def classify(self, C_it):
        """
        function classifies points in X to clusters defined by centers C_it (2 centers assumed).
        It returns the index of the first point classified to cluster 1.
        If no point is classified to cluster 1, it returns n_samples.

        :param C_it: centers at iteration it (shape (2,))
        :return: index of the first point classified to cluster 1, or n_samples if none found
        """
        n_samples = self._X.shape[0]
        for i in range(n_samples):
            classification_i = 0 if np.power(self._X[i] - C_it[0], 2) < np.power(self._X[i] - C_it[1], 2) else 1
            if classification_i == 1:
                return i
        return n_samples

    #############################################
    # plot_iteration
    #############################################
    # function is generating the output file of an iteration.
    # It is not required for the functionality of 'update_centers' computation.
    def plot_iteration(self, c2_start_it, C_it, iteration):
        fig = plt.figure(figsize=(13, 6))
        fig.tight_layout()
        # plotting procedure clustering status (of iteration)
        s1 = plt.subplot(1, 2, 1)
        s1.set_title("Clusters at iteration " + str(iteration))
        classifications = np.zeros(self.n, dtype=int)
        classifications[c2_start_it:] = 1
        # s1.scatter(X[:c2_start_it], X[c2_start_it:], c=c2_start_it, s=2, vmin=0, vmax=k)
        s1.scatter(self._X, np.zeros(self.n), c=classifications, s=2, vmin=0, vmax=self.k)
        s1.scatter([C_it[0], C_it[1]], (0, 0), c=list(range(0, self.k)), s=20)
        # s1.scatter(C_it[0], C_it[1], c=list(range(0, k)), s=5, vmin=0, vmax=k)
        # s1.scatter([0, 1], [0, 1], c="white", s=0)
        # output a figure for iteration
        plt.savefig('iteration' + str(iteration) + '.pdf', bbox_inches='tight')

    #############################################
    # update_centers
    #############################################
    # The functions update_centers are updaetting the centers
    def update_centers(self, c2_start_it, C_it):
        # New non-private center j:
        X_c0 = self._X[:c2_start_it]
        X_c1 = self._X[c2_start_it:]
        if np.any(X_c0):  # compute new center as the average
            C_it[0] = X_c0.mean()  # X_C.mean(axis=0)
        else:
            C_it[0] = np.random.uniform(0, X_c1.mean() - 0.001, 1)
        if np.any(X_c1):  # compute new center as the average
            C_it[1] = X_c1.mean()  # X_C.mean(axis=0)
        else:
            C_it[1] = np.random.uniform(X_c0.mean() + 0.001, 1, 1)

    #############################################
    # kmeans_heuristic
    #############################################
    # function executes two procedures for k-means simulanously:
    def kmeans_heuristic(self):
        C_it = self.C_init  # Choose initial cluster centroids randomly with coordinates in [0,1]
        c2_start_it = self.n
        for it in range(self.max_iter + 1):
            # classify X point to k clusters
            c2_start_it = self.classify(C_it)
            # plot current iteration clusters
            # self.plot_iteration(c2_start_it, C_it, it)
            # update centeroids, privately and non-privately
            self.update_centers(c2_start_it, C_it)

        self.c1 = C_it[0]
        self.c2 = C_it[1]
        # self._c2_start = c2_start_it
        self._n1 = c2_start_it
        self._n2 = self.n - c2_start_it
        return self.c1, self.c2

    def get_moved_points_to_c2(self, old_c2_start_it):
        P = self._X[self._n1:old_c2_start_it]
        n_p = P.shape[0]
        s_p = np.sum(P)
        return P, n_p, s_p

    # function removes the largest point in X
    def delete_largest_points(self, num_deletions=1):
        old_c2_start_it = self._n1
        deleted_points = self._X[-num_deletions:]

        self._X = self._X[:-num_deletions]

        self.n = self.n - num_deletions

        self._n1 = None  # number of points in cluster 1
        self._n2 = None  # number of points in cluster 2
        self.c1 = None  # center of cluster 1
        self.c2 = None  # center of cluster 2
        # print(f"Removed point: {deleted_points} from X")
        c1, c2 = self.kmeans_heuristic()
        self._P, self._n_p, self._s_p = self.get_moved_points_to_c2(old_c2_start_it)

        p2 = np.random.choice(self._P) if self._n_p > 0 else None
        return c1, c2, np.sum(deleted_points), p2
