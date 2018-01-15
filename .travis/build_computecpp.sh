#!/bin/bash

set -ev

###########################
# Get ComputeCpp
###########################
wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/0.5.1/ubuntu-16.04-64bit.tar.gz
tar -xzf ubuntu-14.04-64bit.tar.gz -C /tmp
mv /tmp/ComputeCpp-CE-0.5.1-Ubuntu-14.04-64bit /tmp/computecpp
# Workaround for C99 definition conflict
bash .travis/computecpp_workaround.sh
