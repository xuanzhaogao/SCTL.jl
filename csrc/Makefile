# Builds libsctljl.so, the C-ABI shim over the header-only SCTL library (MPI-free).
#
#   make SCTL_ROOT=/path/to/SCTL [CXX=g++] [EXTRA_CXXFLAGS=...] [EXTRA_LDFLAGS=...] [BUILDDIR=...] [SRCS=...]
#
# Normally driven by deps/build.jl (`Pkg.build("SCTL")`).

SCTL_ROOT ?= ../deps/SCTL
CXX ?= g++
BUILDDIR ?= ../deps/usr
MARCH ?= -march=native

CXXFLAGS = -std=c++17 -fopenmp -fPIC -O3 $(MARCH) -DNDEBUG -Wall -Wno-unused-function -Wno-terminate -fvisibility=hidden -fvisibility-inlines-hidden
CXXFLAGS += -I$(SCTL_ROOT)/include -I.
# AVX-512-FP16 of -march=native (Sapphire Rapids), which binutils before 2.38 can't assemble
# (only where the compiler knows the flag; GCC < 12 does not)
CXXFLAGS += $(shell $(CXX) -mno-avx512fp16 -E -x c++ /dev/null >/dev/null 2>&1 && echo -mno-avx512fp16)
CXXFLAGS += $(EXTRA_CXXFLAGS)
LDFLAGS = -shared -fopenmp $(EXTRA_LDFLAGS)

SRCS ?= $(wildcard capi_*.cpp)
OBJDIR = $(BUILDDIR)/obj
OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))
LIB = $(BUILDDIR)/lib/libsctljl.so

all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@.tmp && mv $@.tmp $@

$(OBJDIR)/%.o: %.cpp $(wildcard *.hpp)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(LIB)

.PHONY: all clean
