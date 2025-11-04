from email.policy import default
from time import time
import numpy as np
import os
import sys
import argparse
import logging
from k_means import KMeansHeuristic
from utils import setup_logger

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


def run_simulations(n_simulations, num_deletions, disc_acc, n, uniform_dist, unif_range_min, unif_range_max,
                    bimodal_cluster1_mean, bimodal_cluster1_std, bimodal_cluster2_mean,
                    bimodal_cluster2_std, bimodal_cluster1_ratio, logger):
    """
    Run k-means attack simulations.
    :param disc_acc: accuracy of the discritization of the domain
    :param uniform_dist: use the uniform distribution (if false use bimodal distribution)
    :param unif_range_min: minimum value of the range.
    :param range_mas: maximum value of the range.
    :param n: number of points to sample from the range in each simulation.
    :param n_simulations:  number of simulations to run.
    :param num_deletions: number of largest points to delete by adversary.
    :param bimodal_cluster1_mean: mean of first cluster
    :param bimodal_cluster1_std: standard deviation of first cluster
    :param bimodal_cluster2_mean: mean of second cluster
    :param bimodal_cluster2_std: standard deviation of second cluster
    :param bimodal_cluster1_ratio: ratio of points in cluster 1 (between 0 and 1)
    :return: None
    """
    global total_clustering_failed_counts
    global total_int_failed_counts
    global total_successful_attacks

    for sim_num in range(n_simulations):
        # randomly select n points with replacement
        km = KMeansHeuristic(disc_acc, uniform_dist, unif_range_min, unif_range_max, n, bimodal_cluster1_mean,
                             bimodal_cluster1_std, bimodal_cluster2_mean, bimodal_cluster2_std,
                             bimodal_cluster1_ratio)
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
            logger.info(
                f"Simulation {sim_num + 1} completed, {total_successful_attacks} successful attacks so far ({total_successful_attacks * 100 / (sim_num + 1):.2f}%).")
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
                    print(f"Found distinct valid pair: (n11={n11}, n_p={n_p}) with real parameters:"
                          f"c11={c11}, c12={c12},c21={c21}, c22={c22}, "
                          f"n={n}, n11={n11}, n12={n - n11} "
                          f"n_p={km._n_p}, S_p={km._s_p}, "
                          f"sample: {km._X} and group P: {km._P}")

        if n11_int_n11_invalid_count > 0 or n11_int_n_p_invalid_count > 0 or len(results) > 1 or len(results) == 0:
            total_int_failed_counts += 1
        else:
            total_successful_attacks += 1

        logger.info(
            f"Simulation {sim_num + 1} completed, {total_successful_attacks} successful attacks so far ({total_successful_attacks * 100 / (sim_num + 1):.2f}%).")


def log_params(args, parser, logger):
    cmd = ["python k_means_simulations.py"]
    for arg, value in vars(args).items():
        default = parser.get_default(arg)

        # For booleans
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg.replace('_', '-')}")  # include flag if True
            # skip if False (since absence implies False)
        else:
            # Use the user-provided value if changed, else default
            cmd.append(f"--{arg.replace('_', '-')} {value}")

    logger.info(" ".join(cmd))


def main():
    parser = argparse.ArgumentParser(description="Run k-means attack simulations.")
    parser.add_argument("--n-simulations", type=int, default=10000, help="Number of simulations to run.")
    parser.add_argument("--num-deletions", type=int, default=50,
                        help="Number of largest points to delete by adversary.")

    parser.add_argument("--disc-acc", type=int, default=2, help="The accurate of discritization of the domain")

    parser.add_argument("--n", type=int, default=500,
                        help="Number of points to sample from the range in each simulation.")

    parser.add_argument("--uniform-dist", action='store_true', help="Whether to use uniform distribution (if not set, use bimodal distribution).")
    parser.add_argument("--unif-range-min", type=float, default=0.0,
                        help="Minimum value of the range for uniform distribution.")
    parser.add_argument("--unif-range-max", type=float, default=1.0,
                        help="Maximum value of the range for uniform distribution.")

    parser.add_argument("--bimodal-cluster1-mean", type=float, default=2.0,
                        help="Mean of first cluster for bimodal distribution.")
    parser.add_argument("--bimodal-cluster1-std", type=float, default=0.5,
                        help="Std dev of first cluster for bimodal distribution.")
    parser.add_argument("--bimodal-cluster2-mean", type=float, default=8.0,
                        help="Mean of second cluster for bimodal distribution.")
    parser.add_argument("--bimodal-cluster2-std", type=float, default=0.5,
                        help="Std dev of second cluster for bimodal distribution.")
    parser.add_argument("--bimodal-cluster1-ratio", type=float, default=0.3,
                        help="Ratio of points in cluster 1 for bimodal distribution.")

    parser.add_argument("--log-file-name", type=str, default=f"log-{int(time())}.txt", help="Name of the log file.")
    args = parser.parse_args()
    setup_logger(output_folder='.', file_name=args.log_file_name, loger_name="k-means-logger",
                 log_level=logging.INFO)

    logger = logging.getLogger("k-means-logger")

    logger.info("======== k-means attack ========")
    log_params(args, parser, logger)

    run_simulations(args.n_simulations, args.num_deletions, args.disc_acc, args.n, args.uniform_dist,
                    args.unif_range_min, args.unif_range_max,
                    args.bimodal_cluster1_mean, args.bimodal_cluster1_std, args.bimodal_cluster2_mean,
                    args.bimodal_cluster2_std,
                    args.bimodal_cluster1_ratio, logger)


if __name__ == "__main__":
    main()

# Recommended command:

# For bimodal distribuiton:
# python k_means_simulations.py --n-simulations 1000 --num-deletions 25 --disc-acc 1 --n 400 --bimodal-cluster1-mean 3.0 --bimodal-cluster1-std 0.4 --bimodal-cluster2-mean 7.0 --bimodal-cluster2-std 0.4 --bimodal-cluster1-ratio 0.45

# For uniform distribuion:
# python k_means_simulations.py --n-simulations 1000 --num-deletions 30 --disc-acc 1 --n 300 --uniform-dist --unif-range-min 0.0 --unif-range-max 10.0