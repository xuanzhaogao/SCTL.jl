// FFT (sctl::FFT<double>): FFTW-backed when built with -DSCTL_HAVE_FFTW, else SCTL's Cooley-Tukey fallback.
//
// Data are row-major (howmany, dims[0], ..., dims[rank-1]) arrays of doubles; complex values are interleaved
// (re, im). R2C/C2R halve the last (fastest) dimension to dims[rank-1]/2+1. Unnormalized; forward e^{-2pi i kn/N}.
#include "sctljl.hpp"

using namespace sctljl;

// type: 0 = R2C, 1 = C2C, 2 = C2C_INV, 3 = C2R (sctl::FFT_Type order)
SCTLJL_EXPORT int sctljl_fft_new(void** out, int type, const int64_t* dims, int64_t rank, int64_t howmany, int nthreads) {
  SCTLJL_TRY
  *out = nullptr;
  if (type < 0 || type > 3) throw Error("sctljl: invalid FFT type");
  if (rank < 1) throw Error("sctljl: FFT rank must be positive");
  if (howmany < 1) throw Error("sctljl: FFT howmany must be positive");
  VecL dim_vec(rank);
  for (Long k = 0; k < rank; k++) {
    if (dims[k] < 1) throw Error("sctljl: FFT dimensions must be positive");
    dim_vec[k] = dims[k];
  }
  auto* fft = new sctl::FFT<double>();
  try {
    fft->Setup(static_cast<sctl::FFT_Type>(type), howmany, dim_vec, nthreads < 1 ? 1 : nthreads);
  } catch (...) {
    delete fft;
    throw;
  }
  *out = fft;
  SCTLJL_CATCH
}

SCTLJL_EXPORT void sctljl_fft_free(void* h) { delete static_cast<sctl::FFT<double>*>(h); }

// Number of doubles in the input (i = 0) and output (i = 1).
SCTLJL_EXPORT int64_t sctljl_fft_dim(void* h, int i) { return static_cast<sctl::FFT<double>*>(h)->Dim(i); }

// in: Dim(0) doubles; out: Dim(1) doubles (caller buffers, any alignment).
SCTLJL_EXPORT int sctljl_fft_execute(void* h, const double* in, int64_t n_in, double* out, int64_t n_out) {
  SCTLJL_TRY
  const auto& fft = *static_cast<sctl::FFT<double>*>(h);
  if (n_in != fft.Dim(0)) throw Error("sctljl: FFT input size mismatch");
  if (n_out != fft.Dim(1)) throw Error("sctljl: FFT output size mismatch");
#ifdef SCTL_HAVE_FFTW
  // FFTW plans assert the buffers' alignment matches the plan's: go through owned (aligned) vectors.
  VecD in_(n_in), out_(n_out);
  for (Long i = 0; i < n_in; i++) in_[i] = in[i];
  fft.Execute(in_, out_, false);  // in_ is our copy: let FFTW destroy it
  copy_to(out, out_, n_out);
#else
  // The Cooley-Tukey fallback never writes its input and writes into an output of the right size in place.
  const VecD in_ = view(in, n_in);
  VecD out_ = view(static_cast<const double*>(out), n_out);
  fft.Execute(in_, out_);
  if (out_.Dim() != n_out || (n_out && &out_[0] != out)) throw Error("sctljl: FFT output was reallocated");
#endif
  SCTLJL_CATCH
}
