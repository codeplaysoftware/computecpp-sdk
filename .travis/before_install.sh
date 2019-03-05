#!/bin/bash

set -ev
sudo apt-get update -q
sudo apt-get install ocl-icd-opencl-dev opencl-headers -y
