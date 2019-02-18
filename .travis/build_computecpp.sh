#!/bin/bash

set -ev

# Get ComputeCpp
wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/1.0.5/ubuntu-14.04-64bit.tar.gz
mkdir -p /tmp/computecpp
tar -xzf ubuntu-14.04-64bit.tar.gz -C /tmp/computecpp --strip-components=1
