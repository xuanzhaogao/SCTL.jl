# Low-level plumbing: library path, error handling, owned-vector handles, callbacks.

using Libdl

const _default_lib = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib", "libsctljl." * Libdl.dlext))
const libsctljl = get(ENV, "SCTL_JL_LIB", _default_lib)

"""
    SCTLError(msg)

An error raised inside the SCTL C++ library (a failed SCTL assertion or `SCTL_ERROR`).
"""
struct SCTLError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SCTLError) = print(io, "SCTLError: ", e.msg)

# State shared with a C callback: the Julia function and the first exception it raised.
mutable struct CallbackState{F}
    f::F
    err::Any
end
CallbackState(f) = CallbackState(f, nothing)

const STATUS_OK = Cint(0)
const STATUS_ERR = Cint(1)
const STATUS_ERR_CALLBACK = Cint(2)

# Turn a status code from the C API into a Julia exception.
function _check(status::Cint, cb::Union{Nothing,CallbackState}=nothing)
    status == STATUS_OK && return nothing
    if status == STATUS_ERR_CALLBACK && cb !== nothing && cb.err !== nothing
        err = cb.err
        cb.err = nothing
        throw(err)
    end
    msg = unsafe_string(ccall((:sctljl_last_error, libsctljl), Cstring, ()))
    throw(SCTLError(status == STATUS_ERR_CALLBACK ? "callback failed: $msg" : msg))
end

# Run a callback body; on an exception record it in the state and return `false`.
@inline function _guard(body, st::CallbackState)
    try
        body()
        return true
    catch e
        st.err === nothing && (st.err = e)
        return false
    end
end

# ---- owned sctl::Vector handles ----

_vsuffix(::Type{Float64}) = "f64"
_vsuffix(::Type{Int64}) = "i64"

_vnew(::Type{Float64}) = ccall((:sctljl_vec_new_f64, libsctljl), Ptr{Cvoid}, ())
_vnew(::Type{Int64}) = ccall((:sctljl_vec_new_i64, libsctljl), Ptr{Cvoid}, ())
_vfree(::Type{Float64}, h) = ccall((:sctljl_vec_free_f64, libsctljl), Cvoid, (Ptr{Cvoid},), h)
_vfree(::Type{Int64}, h) = ccall((:sctljl_vec_free_i64, libsctljl), Cvoid, (Ptr{Cvoid},), h)
_vsize(::Type{Float64}, h) = ccall((:sctljl_vec_size_f64, libsctljl), Int64, (Ptr{Cvoid},), h)
_vsize(::Type{Int64}, h) = ccall((:sctljl_vec_size_i64, libsctljl), Int64, (Ptr{Cvoid},), h)
_vdata(::Type{Float64}, h) = ccall((:sctljl_vec_data_f64, libsctljl), Ptr{Float64}, (Ptr{Cvoid},), h)
_vdata(::Type{Int64}, h) = ccall((:sctljl_vec_data_i64, libsctljl), Ptr{Int64}, (Ptr{Cvoid},), h)
_vresize(::Type{Float64}, h, n) = _check(ccall((:sctljl_vec_resize_f64, libsctljl), Cint, (Ptr{Cvoid}, Int64), h, n))
_vresize(::Type{Int64}, h, n) = _check(ccall((:sctljl_vec_resize_i64, libsctljl), Cint, (Ptr{Cvoid}, Int64), h, n))

"""
    VecHandle{T}

An owned `sctl::Vector<T>` (T = Float64 or Int64) allocated by the shim, used for outputs whose size is only known
after the call. Free with `free!`; usually managed by `_withvecs`.
"""
struct VecHandle{T}
    ptr::Ptr{Cvoid}
end
VecHandle{T}() where {T} = VecHandle{T}(_vnew(T))
Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::VecHandle) = h.ptr
Base.length(h::VecHandle{T}) where {T} = _vsize(T, h.ptr)
free!(h::VecHandle{T}) where {T} = _vfree(T, h.ptr)

# Copy the handle's contents into a new Julia Vector.
function Base.collect(h::VecHandle{T}) where {T}
    n = length(h)
    v = Vector{T}(undef, n)
    n > 0 && unsafe_copyto!(pointer(v), _vdata(T, h.ptr), n)
    return v
end

# Overwrite the handle's contents with `src` (used from callbacks).
function Base.copyto!(h::VecHandle{T}, src::AbstractVector) where {T}
    n = length(src)
    _vresize(T, h.ptr, n)
    s = convert(Vector{T}, src)
    GC.@preserve s (n > 0 && unsafe_copyto!(_vdata(T, h.ptr), pointer(s), n))
    return h
end

# Allocate one handle per element type in `types`, call f(handles...), free them.
function _withvecs(f, types::Type...)
    hs = map(T -> VecHandle{T}(), types)
    try
        return f(hs...)
    finally
        foreach(free!, hs)
    end
end

# Contiguous Float64 / Int64 copies (or the array itself when already suitable) for passing to C.
_f64(x::Array{Float64}) = x
_f64(x::AbstractArray) = convert(Array{Float64}, x)
_f64(x::Real) = Float64(x)
_i64(x::Array{Int64}) = x
_i64(x::AbstractArray) = convert(Array{Int64}, x)

# ---- vectors handed to Julia callbacks ----

# Below this length a callback gets reused Julia buffers (a copy is cheaper than wrapping), above it zero-copy views
# of SCTL's memory (two small array allocations per call, but no O(n) copies).
const CALLBACK_COPY_MAX = 512

mutable struct CallbackBufs
    a::Vector{Float64}
    b::Vector{Float64}
end
CallbackBufs(n::Integer) = n <= CALLBACK_COPY_MAX ? CallbackBufs(zeros(n), zeros(n)) : CallbackBufs(Float64[], Float64[])

# Call g(out, in) on length-n arrays standing for the C buffers `out` (written) and `in` (read).
@inline function _with_cbufs(g, bufs::CallbackBufs, out::Ptr{Float64}, in::Ptr{Float64}, n::Integer)
    if n <= CALLBACK_COPY_MAX
        unsafe_copyto!(pointer(bufs.b), in, n)
        g(bufs.a, bufs.b)
        unsafe_copyto!(out, pointer(bufs.a), n)
    else
        g(unsafe_wrap(Array, out, n), unsafe_wrap(Array, in, n))
    end
    return nothing
end

# ---- OpenMP / build info ----

"""
    SCTL.get_num_threads()

Maximum number of OpenMP threads SCTL will use.
"""
get_num_threads() = Int(ccall((:sctljl_get_max_threads, libsctljl), Cint, ()))

"""
    SCTL.set_num_threads(n)

Set the number of OpenMP threads SCTL uses (`omp_set_num_threads`).
"""
set_num_threads(n::Integer) = ccall((:sctljl_set_num_threads, libsctljl), Cvoid, (Cint,), n)

"""
    SCTL.has_feature(name)

Whether the compiled library has an optional dependency enabled; `name` is one of `:BLAS`, `:LAPACK`, `:FFTW`, `:MPI`.
"""
has_feature(name) = ccall((:sctljl_have, libsctljl), Cint, (Cstring,), string(name)) != 0

function _check_lib()
    isfile(libsctljl) || error("""
        SCTL.jl: shared library not found at $libsctljl.
        Build it with `using Pkg; Pkg.build("SCTL")` (set ENV["SCTL_ROOT"] to the SCTL checkout if needed).""")
end
