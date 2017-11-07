#!/bin/bash

cat .travis/additional_undef /tmp/ComputeCpp-CE-0.3.3-Ubuntu.14.04-64bit/include/SYCL/sycl_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-0.3.3-Ubuntu.14.04-64bit/include/SYCL/sycl_builtins.h

cat .travis/additional_undef /tmp/ComputeCpp-CE-0.3.3-Ubuntu.14.04-64bit/include/SYCL/host_relational_builtins.h > /tmp/tmp_builtins
mv /tmp/tmp_builtins /tmp/ComputeCpp-CE-0.3.3-Ubuntu.14.04-64bit/include/SYCL/host_relational_builtins.h
