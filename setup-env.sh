#!/bin/bash
# basic
mamba create -n Health-LLM python=3.11 -y
mamba activate Health-LLM
mamba install -n Health-LLM cuda-libraries=11.8 cuda-libraries-dev=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 \
                      cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 \
                      cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 cuda-version=12.4 \
                      libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 \
                      -c nvidia -c pytorch -y
