# Advanced Dimensionality Reduction with Deep Learning and t-SNE

This repository contains a Python script that performs advanced dimensionality reduction using a deep learning-based autoencoder combined with t-SNE. After reducing the dimensionality, it applies clustering (e.g., K-Means) on the latent space and plots the results.

## Requirements

To run the script, you need the following Python packages:

- numpy
- matplotlib
- scikit-learn
- tensorflow

## Inputs and Outputs
## Inputs:

X: The input data (e.g., MNIST digits dataset).

y: The target labels (for visualization purposes).

latent_dim: The dimensionality of the latent space (default: 32).

tsne_dim: The dimensionality of the t-SNE output (default: 2).

n_clusters: The number of clusters for K-Means (default: 10).

## Outputs:

X_embedded: The 2D representation of the data after applying autoencoder and t-SNE.

clusters: The cluster labels assigned by K-Means.

A 2D scatter plot showing the clusters in the reduced latent space.

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
