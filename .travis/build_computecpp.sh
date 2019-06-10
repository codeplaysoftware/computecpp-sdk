#!/bin/bash

set -ev
wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/1.1.3/ubuntu-16.04-64bit.tar.gz
mkdir -p /tmp/computecpp
tar -xzf ubuntu-16.04-64bit.tar.gz -C /tmp/computecpp --strip-components=1
