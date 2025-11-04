import numpy as np
from k_means import KMeansHeuristic

n_simulations = 10000  # number of simulations
N = 1000  # number of points in the range
rng = np.linspace(0, 1, N)  # 1,000 equally spaced points
n = 500  # number of points to sample from the range in each simulation
num_deletions = 100  # number of largest points to delete by adversary

for _ in range(n_simulations):
    # randomly select n points with replacement
    C_init = np.sort(np.random.rand(2))  # Choose initial cluster centroids randomly with coordinates in [0,1]

    km = KMeansHeuristic(rng, C_init, n, 2, max_iter=10)
    c11, c12 = km.kmeans_heuristic()
    print(f"Centroids before deletions: {c11}, {c12}")
    c21, c22, p1, p2 = km.delete_largest_points(num_deletions)
    print(f"Centroids after deletion: {c21}, {c22}\n")
