# EDKCD: Edge-Decoupled Koopman Causal Discovery from Time Series in Nonlinear Dynamical Systems

## Introduction

Discovering causal relationships from multivariate time series is essential for understanding nonlinear dynamical systems, yet most learning-based methods rely on scalar weights and lack fine-grained edge analysis. We propose EDKCD, an edge-decoupled Koopman causal discovery method that learns an independent linear transition operator per ordered variable pair, transforming nonlinear causal discovery into decoupled edge-structure learning in a linear latent space. Beyond the Frobenius norm causal strength, EDKCD applies SVD to each operator to analyze multi-channel couplings, multi-lag dependencies, and potential low-rank structures within causal edges. Experiments on synthetic and real-world datasets show that EDKCD effectively recovers Granger causal structures and supports spectral analysis of edge-level dynamics.

## Requirements

- Python 3.11
- See `requirements.txt` for dependencies.

## Usage

```bash
cd bin

# Run with config file
python edkcd.py --config ../config/lorenz96_0.yaml


