#!/bin/bash

set -ev

sudo apt-get update -q
sudo apt-get install ocl-icd-libopencl1 ocl-icd-dev opencl-headers -y
# Use gcc 5 as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
