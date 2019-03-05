#!/bin/bash

set -ev

wget --no-check-certificate https://computecpp.codeplay.com/downloads/computecpp-ce/1.1.0/ubuntu-14.04-64bit.tar.gz
mkdir -p /tmp/computecpp
tar -xzf ubuntu-14.04-64bit.tar.gz -C /tmp/computecpp --strip-components=1
