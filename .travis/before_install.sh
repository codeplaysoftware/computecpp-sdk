#!/bin/bash

set -ev
sudo apt-get update -q
sudo apt-get install ocl-icd-libopencl1 ocl-icd-dev opencl-headers -y
