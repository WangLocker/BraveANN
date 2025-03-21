# RKM

C++ code for the paper "BraveANN: Robust Approximate Nearest Neighbor Search for Billion-Scale Vectors."

## Introduction

This repo holds the source code and scripts for reproducing the key experiments of our paper: **BraveANN: Robust Approximate Nearest Neighbor Search for Billion-Scale Vectors**.

## Datasets

Download the following datasets, and run our `data_process.py` to get the data format that can be used in our code:

| Datasets         | Description                                                                                         | Source                                                                                           |
|------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| DEEP1B           | A large-scale dataset consisting of deep embeddings for billions of images.                          | [Link to DEEP1B](http://sites.skoltech.ru/compvision/noimi/)                                                           |
| SPACEV1B         | Contains vector representations from a large collection of space images.                           | [Link to SPACEV1B](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B)                                                         |
| SIFT1B           | A dataset for testing on large-scale SIFT feature vectors (1 billion).                               | [Link to SIFT1B](http://corpus-texmex.irisa.fr/)                                                           |
| MNIST            | A classic dataset of handwritten digits used for image recognition.                                | [Link to MNIST](https://ieeexplore.ieee.org/document/726791)                                              |
| FASHION-MNIST    | A dataset similar to MNIST but with fashion-related items.                                          | [Link to FASHION-MNIST](https://arxiv.org/abs/1708.07747)                       |
| GIST             | A dataset containing global image descriptors (GIST features).                                       | [Link to GIST](http://corpus-texmex.irisa.fr/)                     |
| SIFT1M           | A subset of the SIFT dataset consisting of 1 million SIFT features.                                 | [Link to SIFT1M](http://corpus-texmex.irisa.fr/)                                |
| SIFT10K          | A larger version of the SIFT dataset consisting of 10 million SIFT features.                        | [Link to SIFT10K](http://corpus-texmex.irisa.fr/)                                |

## Build (C++ Version)

- swig >= 4.0.2
- cmake >= 3.12.0
- boost >= 1.67.0
- xtensor >= 0.24.0

## Install & Run (C++ Version)

```bash
mkdir build
cd build && cmake .. && make
```
Then you can config and run the build and search scripts according to the samples in the repo. 

## Install (Python Version)

```bash
pip install numpy pandas scikit-learn
```

## Run (Python Version)
Just run the main.py