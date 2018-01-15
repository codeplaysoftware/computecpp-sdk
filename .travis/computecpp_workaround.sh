#!/bin/bash

cat .travis/additional_undef /tmp/computecpp/include/SYCL/sycl_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/computecpp/include/SYCL/sycl_builtins.h

cat .travis/additional_undef /tmp/computecpp/include/SYCL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/computecpp/include/SYCL/host_relational_builtins.h
