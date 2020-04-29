#!/bin/bash

set -ev
wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/2.0.0/x86_64-linux-gnu.tar.gz
mkdir -p /tmp/computecpp
tar -xzf x86_64-linux-gnu.tar.gz -C /tmp/computecpp --strip-components=1
