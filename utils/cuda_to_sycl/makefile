# Your ComputeCpp installation root.
COMPUTECPP_PREFIX ?= /usr/local

COMPUTECPP ?= $(COMPUTECPP_PREFIX)/bin/compute++
COMPUTECPP_INFO ?= $(COMPUTECPP_PREFIX)/bin/computecpp_info
COMPUTECPP_INCLUDES ?= $(COMPUTECPP_PREFIX)/include
COMPUTECPP_LIBS ?= $(COMPUTECPP_PREFIX)/lib

NVCC=nvcc

# In addition your normal flags, compilation requires C++11 standard,
# the SYCL headers, and the ComputeCpp library.
CXXFLAGS += --std=c++11 -I$(COMPUTECPP_INCLUDES)
LDFLAGS += -L$(COMPUTECPP_LIBS) -lComputeCpp
COMPUTECPP_FLAGS += \
	$(CXXFLAGS) $(shell $(COMPUTECPP_INFO) --dump-device-compiler-flags)

all: SYCLversion CUDAversion

# Single source multiple pass compilation.
SYCLversion: SYCLversion.cpp
	$(COMPUTECPP) $(COMPUTECPP_FLAGS) -c $< -o $@.sycl
	$(CXX) $(CXXFLAGS) -include $@.sycl $< -o $@ $(LDFLAGS)

CUDAversion: CUDAversion.cu
	$(NVCC) $<

.PHONY: clean help
clean:
	rm -fv SYCLversion SYCLversion.sycl CUDAversion

help:
	@echo "Usage:  make COMPUTECPP_PREFIX=[path-to-computecpp] {all,clean,help}"
