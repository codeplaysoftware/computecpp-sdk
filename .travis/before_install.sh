#!/bin/bash

set -ev

# Add modern dependencies (cmake, boost)
sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
sudo add-apt-repository ppa:kzemek/boost -y
sudo apt-get update -q
sudo apt-get install cmake -y
# TriSYCL requires modern boost
sudo apt-get install boost1.58 -y
# Use gcc 5 as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
# Install the khronos stub opencl
wget https://github.com/KhronosGroup/OpenCL-Headers/archive/master.zip -O /tmp/ocl-headers.zip
unzip /tmp/ocl-headers.zip -d /tmp
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git /tmp/OpenCL-ICD-Loader
ln -sf /tmp/OpenCL-Headers-master/opencl20/CL /tmp/OpenCL-ICD-Loader/inc/CL
pushd /tmp/OpenCL-ICD-Loader && git checkout cb4acb99df3441e147bd859c20d9c3533ec95ce8 && make && popd
# Recreate a fake OpenCL setup
sudo mkdir /usr/include/CL/
sudo cp -rf /tmp/OpenCL-Headers-master/opencl20/CL /usr/include/CL/
sudo cp -Rf /tmp/OpenCL-ICD-Loader/build/bin/* /usr/lib/
sudo cp -Rf /tmp/OpenCL-ICD-Loader/build/bin/* /usr/lib/x86_64-linux-gnu/
