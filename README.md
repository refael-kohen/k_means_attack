# k-means-attack

A Python project that simulates an adversarial k-means attack on one-dimensional datasets. The code generates synthetic datasets (uniform or bimodal Gaussian mixtures), runs a k-means heuristic for two clusters, simulates deletion of the largest points by an adversary, and attempts to reconstruct cluster membership by solving the equations via integer substitution.

---

## Features

- Generate 1-D datasets using:
  - Uniform distribution
  - Bimodal Gaussian mixture (two clusters)
- k-means heuristic (two clusters) with discretized values
- Simulate an adversary that deletes the top `k` largest points
- Analyze how many points move clusters after deletion and attempt reconstruction of integer parameters
- A log file is created (log-<timestamp>.txt) containing informational messages and the final aggregated statistics.

## Repository layout

- `k_means.py` - core algorithm: `KMeansHeuristic` class implementing dataset sampling, k-means classification, deletion of largest points, and helpers.
- `k_means_simulations.py` - command-line simulation driver: runs many trials and aggregates statistics.
- `utils.py` - helper utilities (logging setup, etc.).

## Requirements

The project is written for Python 3.8+ and depends on:

- numpy


## CLI reference

All command-line options are exposed by `k_means_simulations.py` (the long form shown here):

- `--n-simulations` (int, default `10000`) - Number of simulations to run.
- `--num-deletions` (int, default `50`) - Number of largest points deleted by adversary.
- `--disc-acc` (int, default `2`) - Decimal places used to round / discretize values.
- `--n` (int, default `500`) - Number of points to sample per simulation.
- `--uniform-dist` (flag) - Use uniform distribution; omit to use bimodal.
- `--unif-range-min` (float, default `0.0`) - Minimum value for uniform distribution.
- `--unif-range-max` (float, default `1.0`) - Maximum value for uniform distribution.
- `--bimodal-cluster1-mean` (float, default `2.0`)
- `--bimodal-cluster1-std` (float, default `0.5`)
- `--bimodal-cluster2-mean` (float, default `8.0`)
- `--bimodal-cluster2-std` (float, default `0.5`)
- `--bimodal-cluster1-ratio` (float, default `0.3`) - Fraction of points in cluster1.
- `--log-file-name` (string, default `log-<timestamp>.txt`) - Log file name.


## Examples

Bimodal example (recommended for experiments where clusters overlap moderately):

```bash
python k_means_simulations.py  \
  --n-simulations 1000 \
  --num-deletions 20 \
  --disc-acc 2 \
  --n 500 \
  --unif-range-min 0.0 \
  --unif-range-max 1.0 \
  --bimodal-cluster1-mean 0.4 \ 
  --bimodal-cluster1-std 0.15 \
  --bimodal-cluster2-mean 0.6 \
  --bimodal-cluster2-std 0.15 \
  --bimodal-cluster1-ratio 0.4 \
  --log-file-name log-bimodal-run.txt
 ```

Uniform example:

```bash
python k_means_simulations.py \
  --n-simulations 1000 \
  --num-deletions 20 \
  --disc-acc 2 \
  --n 500 \
  --uniform-dist \
  --unif-range-min 0.0 \
  --unif-range-max 1.0 \
  --bimodal-cluster1-mean 2.0 \
  --bimodal-cluster1-std 0.5 \
  --bimodal-cluster2-mean 8.0 \
  --bimodal-cluster2-std 0.5 \
  --bimodal-cluster1-ratio 0.3 \
  --log-file-name log-uniform-run.txt
```

The driver writes an execution log file (by default `log-<timestamp>.txt`) in the working directory and prints progress messages via the logger. All aggregated simulation statistics are appended at the end of the log file and also printed to the command line.