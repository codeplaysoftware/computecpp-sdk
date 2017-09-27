#!/bin/bash

set -ev

###########################
# Get ComputeCpp
###########################
wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/0.3.2/ubuntu-14.04-64bit.tar.gz
tar -xzf ComputeCpp-CE-0.3.2-Ubuntu-14.04-64bit.tar.gz -C /tmp
# Workaround for C99 definition conflict
bash .travis/computecpp_workaround.sh
