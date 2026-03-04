#!/bin/bash

conda create -n gegd_dev python=3.12
conda activate gegd_dev
conda install -c conda-forge numpy scipy mpi4py numba matplotlib psutil
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade fmmax ceviche