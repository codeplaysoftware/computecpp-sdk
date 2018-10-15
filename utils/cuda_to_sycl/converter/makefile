# Your ComputeCpp installation root.
COMPUTECPP_DIR ?= /usr/local

COMPUTECPP ?= $(COMPUTECPP_DIR)/bin/compute++
COMPUTECPP_INFO ?= $(COMPUTECPP_DIR)/bin/computecpp_info
COMPUTECPP_INCLUDES ?= $(COMPUTECPP_DIR)/include
COMPUTECPP_LIBS ?= $(COMPUTECPP_DIR)/lib
COMPUTECPP_SDK_INCLUDES ?= ../../../include
NVCC=nvcc

# In addition your normal flags, compilation requires C++11 standard,
# the SYCL headers, and the ComputeCpp library.
CXXFLAGS += --std=c++11 -I$(COMPUTECPP_INCLUDES) -I$(COMPUTECPP_SDK_INCLUDES) -I.
LDFLAGS += -L$(COMPUTECPP_LIBS) -lComputeCpp  -lOpenCL

COMPUTECPP_FLAGS += \
	-sycl-driver -no-serial-memop -mllvm -inline-threshold=1000  $(CXXFLAGS) 

all: add_stride add

# Single source multiple pass compilation.
add_stride: add_stride.cpp
	$(COMPUTECPP) $(COMPUTECPP_FLAGS) $^ -o $@ $(LDFLAGS)
# Single source multiple pass compilation.
add: add.cpp
	$(COMPUTECPP) $(COMPUTECPP_FLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: clean help
clean:
	rm -fv add_stride add 

help:
	@echo "Usage:  make COMPUTECPP_DIR=[path-to-computecpp] {all,clean,help}"
