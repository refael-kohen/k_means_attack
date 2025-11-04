import numpy as np
import matplotlib.pyplot as plt


class KMeansHeuristic:
    def __init__(self, disc_acc, uniform_dist, unif_range_min, unif_range_max, n, bimodal_cluster1_mean=None,
                 bimodal_cluster1_std=None, bimodal_cluster2_mean=None,
                 bimodal_cluster2_std=None, bimodal_cluster1_ratio=None, max_iter=10):

        self.uniform_dist = uniform_dist
        self.unif_range_max = unif_range_max
        self.unif_range_min = unif_range_min
        self.bimodal_cluster1_mean = bimodal_cluster1_mean
        self.bimodal_cluster2_mean = bimodal_cluster2_mean
        self.bimodal_cluster1_std = bimodal_cluster1_std
        self.bimodal_cluster2_std = bimodal_cluster2_std
        self.bimodal_cluster1_ratio = bimodal_cluster1_ratio
        self.disc_acc = disc_acc
        self.n = n  # number of data points
        self.max_iter = max_iter
        self.c1 = None  # center of cluster 1 (left) - retrain it after deletion
        self.c2 = None  # center of cluster 2 (right) - retrain it after deletion
        self._n1 = None  # number of points in cluster 1
        self._n2 = None  # number of points in cluster 2
        self._P = None  # points moved to cluster 2 after deletions
        self._n_p = None  # number of points moved to cluster 2 after deletions
        self._s_p = None  # sum of points moved to cluster 2 after deletions
        self._s1 = None  # sum of points in cluster 1 - for numeric accuracy
        self._s2 = None  # sum of points in cluster 2 - for numeric accuracy
        self.C = None  # retrain it after deletion
        self._X = self.sample_dataset()

    def pick_init_cetroids(self):
        if self.uniform_dist:
            centroids = np.sort(np.random.rand(2))  # Choose initial cluster centroids randomly
            return centroids[0], centroids[1]
        else:
            centroids = np.sort(np.random.rand(2)) * (
                    self.bimodal_cluster2_mean - self.bimodal_cluster1_mean) + self.bimodal_cluster1_mean
            return centroids[0], centroids[1]

    def sample_dataset(self):
        if self.uniform_dist:
            return np.sort(
                np.round(np.random.uniform(self.unif_range_min, self.unif_range_max, self.n), decimals=self.disc_acc))
        else:
            # Generate bimodal data: two Gaussian clusters
            n1 = int(self.n * self.bimodal_cluster1_ratio)
            n2 = self.n - n1
            cluster1_data = np.random.normal(self.bimodal_cluster1_mean, self.bimodal_cluster1_std, n1)
            cluster2_data = np.random.normal(self.bimodal_cluster2_mean, self.bimodal_cluster2_std, n2)
            return np.sort(np.round(np.concatenate([cluster1_data, cluster2_data]), decimals=self.disc_acc))

    #############################################
    # classify
    #############################################
    def classify(self):
        n_samples = self._X.shape[0]
        for i in range(n_samples):
            # Calculate squared distances without np.power (more numeric stable)
            dist_0 = (self._X[i] - self.c1) * (self._X[i] - self.c1)
            dist_1 = (self._X[i] - self.c2) * (self._X[i] - self.c2)
            classification_i = 0 if dist_0 < dist_1 else 1
            if classification_i == 1:
                return i
        return n_samples

    #############################################
    # plot_iteration
    #############################################
    # function is generating the output file of an iteration.
    # It is not required for the functionality of 'update_centers' computation.
    def plot_iteration(self, c2_start_it, iteration):
        fig = plt.figure(figsize=(13, 6))
        fig.tight_layout()
        # plotting procedure clustering status (of iteration)
        s1 = plt.subplot(1, 2, 1)
        s1.set_title("Clusters at iteration " + str(iteration))
        classifications = np.zeros(self.n, dtype=int)
        classifications[c2_start_it:] = 1
        s1.scatter(self._X, np.zeros(self.n), c=classifications, s=2, vmin=0, vmax=2)
        s1.scatter([self.c1, self.c2], (0, 0), c=list(range(0, 2)), s=20)
        # output a figure for iteration
        plt.savefig('iteration' + str(iteration) + '.pdf', bbox_inches='tight')

    #############################################
    # update_centers
    #############################################
    # The functions update_centers are updaetting the centers
    def update_centers(self, c2_start_it):
        # New non-private center j:
        X_c0 = self._X[:c2_start_it]
        X_c1 = self._X[c2_start_it:]
        if np.any(X_c0):  # compute new center as the average
            self._s1 = np.sum(X_c0, dtype=np.float64)  # for numeric accuracy
            self.c1 = self._s1 / np.float64(len(X_c0))
        else:
            if self.uniform_dist:
                self.c1 = np.random.uniform(self.unif_range_min, X_c1.mean() - 0.001, 1)
            else:
                self.c1 = np.random.uniform(X_c1.mean() - 0.001 - 2 * self.bimodal_cluster1_std, X_c1.mean() - 0.001, 1)

        if np.any(X_c1):  # compute new center as the average
            # C_it[1] = X_c1.mean()
            self._s2 = np.sum(X_c1, dtype=np.float64)  # for numeric accuracy
            self.c2 = self._s2 / len(X_c1)
        else:
            if self.uniform_dist:
                self.c2 = np.random.uniform(self.c1 + 0.001, self.unif_range_max, 1)
            else:
                self.c2 = np.random.uniform(self.c1 + 0.001, self.c1 + 0.001 + 2 * self.bimodal_cluster2_std, 1)

    #############################################
    # kmeans_heuristic
    #############################################
    def kmeans_heuristic(self):
        """function executes two procedures for k-means simulanously"""
        self.c1, self.c2 = self.pick_init_cetroids()  # Choose initial cluster centroids randomly
        c2_start_it = self.n
        for it in range(self.max_iter + 1):
            # classify X point to 2 clusters
            c2_start_it = self.classify()
            # plot current iteration clusters
            # self.plot_iteration(c2_start_it, C_it, it)
            # update centeroids, privately and non-privately
            self.update_centers(c2_start_it)

        self._n1 = c2_start_it
        self._n2 = self.n - c2_start_it
        return self.c1, self.c2

    def get_moved_points_to_c2(self, old_c2_start_it):
        # TODO: If num_deletions > n12 need to check what happens. 
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
        _, _ = self.kmeans_heuristic()
        self._P, self._n_p, self._s_p = self.get_moved_points_to_c2(old_c2_start_it)

        p2 = np.random.choice(self._P) if self._n_p > 0 else None
        return self.c1, self.c2, np.sum(deleted_points), p2
