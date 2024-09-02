#!/bin/bash

sudo apt update
sudo apt install linux-headers-$(uname -r)
sudo apt install nvidia-driver firmware-misc-nonfree
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-cuda-toolkit-gcc

export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
