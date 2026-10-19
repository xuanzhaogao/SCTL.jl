# Builds libsctljl.so, the C-ABI shim over the header-only SCTL library (MPI-free).
#
#   make SCTL_ROOT=/path/to/SCTL [PVFMM_ROOT=/path/to/pvfmm] [BLAS=lbt | BLAS_LIBDIR=/dir/with/libopenblas.so]
#        [CXX=g++] [EXTRA_CXXFLAGS=...] [EXTRA_LDFLAGS=...] [BUILDDIR=...] [SRCS=...]
#
# PVFMM_ROOT (a PVFMM checkout; only its headers are used, compiled without MPI) enables the FMM in ParticleFMM and
# BoundaryIntegralOp. BLAS=lbt enables BLAS and LAPACK from Julia's libblastrampoline, set at load time (capi_blas.cpp;
# the default of deps/build.jl); BLAS_LIBDIR instead links an LP64 libopenblas with LAPACK from that directory.
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
LDFLAGS = -shared -fopenmp
ifneq ($(PVFMM_ROOT),)
CXXFLAGS += -DSCTL_HAVE_PVFMM -I$(PVFMM_ROOT)/include
endif
ifeq ($(BLAS),lbt)
CXXFLAGS += -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK -DSCTLJL_BLAS_LBT
else ifneq ($(BLAS_LIBDIR),)
CXXFLAGS += -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK
LDFLAGS += -L$(BLAS_LIBDIR) -lopenblas -Wl,-rpath,$(BLAS_LIBDIR)
endif
CXXFLAGS += $(EXTRA_CXXFLAGS)
LDFLAGS += $(EXTRA_LDFLAGS)

# rebuild everything when the configuration changes
CONFIG = $(BUILDDIR)/obj/config.txt
CONFIG_NOW = $(CXX) $(CXXFLAGS) | $(LDFLAGS)
$(shell mkdir -p $(BUILDDIR)/obj; [ "$$(cat $(CONFIG) 2>/dev/null)" = '$(CONFIG_NOW)' ] || echo '$(CONFIG_NOW)' > $(CONFIG))

SRCS ?= $(wildcard capi_*.cpp)
OBJDIR = $(BUILDDIR)/obj
OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))
LIB = $(BUILDDIR)/lib/libsctljl.so

all: $(LIB)

$(LIB): $(OBJS) $(CONFIG)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@.tmp && mv $@.tmp $@

$(OBJDIR)/%.o: %.cpp $(wildcard *.hpp) $(CONFIG)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(LIB)

.PHONY: all clean
