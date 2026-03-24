#!/bin/bash

conda create -y -n gegd_dev python=3.12 pip
conda install -y -n gegd_dev -c conda-forge numpy scipy mpi4py numba matplotlib psutil
conda run -n gegd_dev python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n gegd_dev python3 -m pip install -U "jax[cuda12]"
conda run -n gegd_dev python3 -m pip install --upgrade fmmax ceviche