# Example Makefile to build a SYCL application using ComputeCpp.

# Your ComputeCpp installation root.
COMPUTECPP_PREFIX ?= /usr/local

COMPUTECPP ?= $(COMPUTECPP_PREFIX)/bin/compute++
COMPUTECPP_INFO ?= $(COMPUTECPP_PREFIX)/bin/computecpp_info
COMPUTECPP_INCLUDES ?= $(COMPUTECPP_PREFIX)/include
COMPUTECPP_LIBS ?= $(COMPUTECPP_PREFIX)/lib

# In addition your normal flags, compilation requires C++11 standard,
# the SYCL headers, and the ComputeCpp library.
CXXFLAGS += --std=c++11 -I$(COMPUTECPP_INCLUDES)
LDFLAGS += -L$(COMPUTECPP_LIBS) -lComputeCpp
COMPUTECPP_FLAGS += \
	$(CXX_FLAGS) $(shell $(COMPUTECPP_INFO) --dump-device-compiler-flags)

# The example application to build.
target := app
all: $(target)

# Single source multiple pass compilation.
%: %.cpp
	$(COMPUTECPP) $(COMPUTECPP_FLAGS) -c $< -o $@.sycl
	$(CXX) $(CXXFLAGS) -include $@.sycl $< -o $@ $(LDFLAGS)


.PHONY: clean help
clean:
	rm -fv $(target) $(target).sycl $(target).o

help:
	@echo "Builds an example SYCL application."
	@echo "Usage:  make COMPUTECPP_PREFIX=[path-to-computecpp] {all,clean,help}"
