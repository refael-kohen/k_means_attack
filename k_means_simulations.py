from email.policy import default

import numpy as np
import os
import sys
import argparse
import logging
from k_means import KMeansHeuristic
from utils import setup_logger

setup_logger(output_folder='.', file_name="log.txt", loger_name="k-means-logger",
             log_level=logging.INFO)

logger = logging.getLogger("k-means-logger")

# Counters for various failed attacks depends on the clustering failures
c21_c22_equals_count = 0
c11_c21_c22_c12_zero_count = 0
_n11_invalid_count = 0
_n12_less_2_count = 0
_np_less_2_count = 0
_n11_zero_count = 0
_n12_zero_count = 0

# Counter for total failed attacks depending
total_clustering_failed_counts = 0
# Counter for total failed attacks depending on integer (n11, n_p) pairs
total_int_failed_counts = 0
# Counter for total successful attacks
total_successful_attacks = 0


def attack_success_validations(c11=None, c12=None, c21=None, c22=None, _n11=None, _n12=None, _np=None, n=None):
    global c21_c22_equals_count
    global c11_c21_c22_c12_zero_count
    global _n11_invalid_count
    global _n12_less_2_count
    global _np_less_2_count
    global _n11_zero_count
    global _n12_zero_count

    if c21 == c22:
        c21_c22_equals_count += 1
        return False
    if c11 - c21 + c22 - c12 == 0:
        c11_c21_c22_c12_zero_count += 1
        return False
    if _n11 == 0 or _n11 > n - 2:
        _n11_invalid_count += 1
        return False
    if _n12 < 2:
        _n12_less_2_count += 1
        return False
    if _np < 2:
        _np_less_2_count += 1
        return False
    return True


def run_simulations(N, range_min, range_max, n, n_simulations, num_deletions):
    """
    Run k-means attack simulations.
    :param N: Number of points in the range.
    :param range_min: minimum value of the range.
    :param range_mas: maximum value of the range.
    :param n: number of points to sample from the range in each simulation.
    :param n_simulations:  number of simulations to run.
    :param num_deletions: number of largest points to delete by adversary.
    :return: None
    """
    global total_clustering_failed_counts
    global total_int_failed_counts
    global total_successful_attacks

    rng = np.linspace(range_min, range_max, N)  # 1,000 equally spaced points

    for sim_num in range(n_simulations):
        # randomly select n points with replacement
        C_init = np.sort(np.random.rand(2))  # Choose initial cluster centroids randomly with coordinates in [0,1]

        km = KMeansHeuristic(rng, C_init, n, 2, max_iter=10)
        c11, c12 = km.kmeans_heuristic()
        logger.info(f"centroids step 1: {c11}, {c12}")
        _n11 = km._n1
        _n12 = km._n2
        c21, c22, p1, p2 = km.delete_largest_points(num_deletions)

        _n_p = km._n_p
        logger.info(f"centroids step 2: {c21}, {c22} p1={p1}, p2={p2}, np={_n_p}")

        # Analyze results
        # Validations of attack success conditions
        valid = attack_success_validations(c11, c12, c21, c22, _n11, _n12, _n_p, n)
        if not valid:
            total_clustering_failed_counts += 1
            continue

        # Calculate possible (n11, n_p) pairs
        results = []
        # Counters for invalid (n11, n_p) pairs
        n11_int_n11_invalid_count = 0
        n11_int_n_p_invalid_count = 0
        for n11 in range(1, n):
            n_p = ((c11 - c21 + c22 - c12) * n11 + (c12 - c22) * n + c21 - p1) / (c21 - c22)
            if n_p.is_integer():
                if n_p < 2 or n_p > n - n11:
                    n11_int_n_p_invalid_count += 1
                elif n11 < 1 or n11 > n - 2:
                    n11_int_n11_invalid_count += 1
                else:
                    results.append((n11, int(n_p)))
                    print(f"Found valid pair: (n11={n11}, n_p={int(n_p)})")

        if n11_int_n11_invalid_count > 0 or n11_int_n_p_invalid_count > 0 or len(results) > 1 or len(results) == 0:
            total_int_failed_counts += 1
        else:
            total_successful_attacks += 1

        logger.info(
            f"Simulation {sim_num + 1} completed, {total_successful_attacks} successful attacks so far ({total_successful_attacks * 100 / (sim_num + 1):.2f}%).")


def log_params(args, parser):
    cmd = ["python k_means_simulations.py"]
    for arg, value in vars(args).items():
        default = parser.get_default(arg)

        # For booleans
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg.replace('_','-')}")  # include flag if True
            # skip if False (since absence implies False)
        else:
            # Use the user-provided value if changed, else default
            cmd.append(f"--{arg.replace('_','-')} {value}")

    logger.info(" ".join(cmd))

def main():
    parser = argparse.ArgumentParser(description="Run k-means attack simulations.")
    parser.add_argument("--range-min", type=float, default=0.0, help="Minimum value of the range.")
    parser.add_argument("--range-max", type=float, default=1.0, help="Maximum value of the range.")
    parser.add_argument("--n-simulations", type=int, default=100, help="Number of simulations to run.")
    parser.add_argument("--N", type=int, default=1000, help="Number of points in the range.")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of points to sample from the range in each simulation.")
    parser.add_argument("--num-deletions", type=int, default=100,
                        help="Number of largest points to delete by adversary.")
    args = parser.parse_args()

    logger.info("======== k-means attack ========")
    log_params(args, parser)
    run_simulations(args.N, args.range_min, args.range_max, args.n, args.n_simulations, args.num_deletions)


if __name__ == "__main__":
    main()
