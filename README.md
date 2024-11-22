# BraveANN
C++ code for paper "BraveANN: Robust Approximate Nearest NeighborSearch for Billion-Scale Vectors"
## Introduction

This repo holds the source code and scripts for reproducing the key experiments of our paper: BraveANN: Robust Approximate Nearest NeighborSearch for Billion-Scale Vectors.

## Datasets

Download the following datasets, and run our `data_process.py`, you can get the data format that can be used in our codes. 

|Datasets|  Description  | Source |
|-------- |------- |-------- |
|SIFT1B |    SIFT1B contains one billion 128-dimensional SIFT vectors, designed to evaluate large-scale ANNS algorithms and their scalability.|  http://corpus-texmex.irisa.fr/  |
|SPACEV1B  | SPACEV1B features one billion high-dimensional vectors, ideal for testing semantic similarity and spatial search performance. |https://learning2hash.github.io/publications/microsoftspacev1B/ |
|DEEP1B  | DEEP1B includes one billion 96-dimensional embeddings from deep learning models, used to benchmark semantic search and indexing techniques.  |   https://sites.skoltech.ru/compvision/noimi/|



## Build
- swig >= 4.0.2
- cmake >= 3.12.0
- boost >= 1.67.0
- xtensor >= 0.24.0

## Install
```bash
mkdir build
cd build && cmake .. && make
```

## Run
After compiling and installing, the release (or debug) directory will contain executable files such as indexbuilder, indexsearcher, and ssdserving. These are used for index construction, performing searches using the index, and conducting overall experiments, respectively. For parameter configuration, ensure that the parameter configuration file is properly set up and specify its path when executing commands in the terminal.

## Info
Our code implementation is based on SPANN. If you encounter issues related to project building, running, or parameter configuration, we kindly recommend referring to its documentation and issues for guidance.