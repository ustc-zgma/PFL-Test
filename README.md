# PFL-Test

## Environment
OS: Ubuntu 20.04

Python version: 3.8.12 [GCC 7.5.0]

Torch version: 1.10.2

CUDA version: 11.3

GPU: NVIDIA GeForce RTX 3090 (24G)

CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz

## Results

Average time for cosine similarity: 0.0136 s

Average time for l^2 norm: 0.0041s

Average time for model pruning and local updating: 0.7011s / 46.1878s

Average time for calculating loss on a mini-batch: 0.0835 s, std: 0.0143, avg_gap:0.0124
