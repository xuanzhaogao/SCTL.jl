# Fast Fourier transforms (sctl::FFT<double>): FFTW when the library is built with -DSCTL_HAVE_FFTW, otherwise
# SCTL's own mixed-radix Cooley-Tukey fallback (any size, any rank, all four transform types).

export FFTPlan, dim, execute, execute!, input_size, output_size

const _FFT_TYPES = (:r2c, :c2c, :c2c_inv, :c2r)  # codes 0..3 = sctl::FFT_Type order

"""
    FFTPlan(type, dims; howmany=1, nthreads=1)

A plan for `howmany` independent `length(dims)`-dimensional discrete Fourier transforms of size `dims`
(`sctl::FFT<double>`). `type` is one of

| `type`     | input                         | output                        | sign |
|:-----------|:------------------------------|:------------------------------|:-----|
| `:r2c`     | real, `dims`                  | complex, `(dims[1]÷2+1, dims[2:end]...)` | `-` |
| `:c2c`     | complex, `dims`               | complex, `dims`               | `-`  |
| `:c2c_inv` | complex, `dims`               | complex, `dims`               | `+`  |
| `:c2r`     | complex, `(dims[1]÷2+1, dims[2:end]...)` | real, `dims`       | `+`  |

**Layout.** `dims` is the size of the Julia (column-major) array being transformed; each of the `howmany`
transforms is a contiguous block, so a batch is a Julia array of size `(dims..., howmany)`. As in FFTW.jl's
`rfft`, the real transforms halve the *first* Julia dimension. (SCTL itself sees row-major arrays of shape
`(howmany, reverse(dims)...)`, halving its last dimension; complex numbers are interleaved `(re, im)` doubles, i.e.
`ComplexF64` memory.)

**Convention.** Forward (`:r2c`, `:c2c`): ``X_k = \\sum_n x_n e^{-2\\pi i\\, k\\cdot n / N}``; inverse
(`:c2c_inv`, `:c2r`): the same with ``e^{+2\\pi i\\, k\\cdot n / N}``. **Unnormalized** both ways, so an inverse after a
forward transform multiplies by `N = prod(dims)`. `:c2r` assumes Hermitian-symmetric input (like FFTW's `c2r`; the
imaginary parts that are inconsistent with a real result are ignored).

`nthreads` is the number of OpenMP threads used by the transform.
"""
mutable struct FFTPlan
    ptr::Ptr{Cvoid}
    type::Symbol
    dims::Vector{Int}
    howmany::Int
    function FFTPlan(type::Symbol, dims; howmany::Integer=1, nthreads::Integer=1)
        code = findfirst(==(type), _FFT_TYPES)
        code === nothing && throw(ArgumentError("FFT type must be one of $(_FFT_TYPES), got :$type"))
        d = Int[dims...]
        isempty(d) && throw(ArgumentError("dims must not be empty"))
        all(>(0), d) || throw(ArgumentError("all dims must be positive, got $(Tuple(d))"))
        howmany > 0 || throw(ArgumentError("howmany must be positive"))
        nthreads > 0 || throw(ArgumentError("nthreads must be positive"))
        sdims = reverse(d)  # SCTL's row-major order
        out = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall((:sctljl_fft_new, libsctljl), Cint, (Ref{Ptr{Cvoid}}, Cint, Ptr{Int64}, Int64, Int64, Cint),
                     out, code - 1, sdims, length(sdims), howmany, nthreads))
        p = new(out[], type, d, Int(howmany))
        finalizer(_free!, p)
        return p
    end
end
FFTPlan(type::Symbol, n::Integer; kw...) = FFTPlan(type, (n,); kw...)

function _free!(p::FFTPlan)
    if p.ptr != C_NULL
        ccall((:sctljl_fft_free, libsctljl), Cvoid, (Ptr{Cvoid},), p.ptr)
        p.ptr = C_NULL
    end
    return nothing
end

"""
    close(plan::FFTPlan)

Release the plan's C++ object now (otherwise done by the finalizer). The plan cannot be used afterwards.
"""
Base.close(p::FFTPlan) = _free!(p)

function _ptr(p::FFTPlan)
    p.ptr == C_NULL && throw(ArgumentError("FFTPlan has been closed"))
    return p.ptr
end

# Passing the object itself to ccall (instead of its raw pointer) keeps it rooted, so the finalizer cannot free the
# C++ object during the call.
Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::FFTPlan) = _ptr(x)

Base.show(io::IO, p::FFTPlan) =
    print(io, "FFTPlan(:", p.type, ", ", Tuple(p.dims), "; howmany=", p.howmany, p.ptr == C_NULL ? ", closed)" : ")")

_half(d) = (d[1] ÷ 2 + 1, d[2:end]...)
_in_complex(p::FFTPlan) = p.type !== :r2c
_out_complex(p::FFTPlan) = p.type !== :c2r

"""
    input_size(plan) -> Tuple

Size of one transform's input (excluding `howmany`): `dims`, or `(dims[1]÷2+1, dims[2:end]...)` for `:c2r`.
"""
input_size(p::FFTPlan) = p.type === :c2r ? _half(p.dims) : Tuple(p.dims)

"""
    output_size(plan) -> Tuple

Size of one transform's output (excluding `howmany`): `dims`, or `(dims[1]÷2+1, dims[2:end]...)` for `:r2c`.
"""
output_size(p::FFTPlan) = p.type === :r2c ? _half(p.dims) : Tuple(p.dims)

"""
    dim(plan, i)

Total number of input (`i = 1`) or output (`i = 2`) elements, over all `howmany` transforms: `Float64`s on a real
side, `ComplexF64`s on a complex side (SCTL's `FFT::Dim(i-1)` counts doubles; this counts Julia elements).
"""
function dim(p::FFTPlan, i::Integer)
    i in (1, 2) || throw(ArgumentError("i must be 1 (input) or 2 (output)"))
    n = ccall((:sctljl_fft_dim, libsctljl), Int64, (Ptr{Cvoid}, Cint), p, i - 1)
    return Int(((i == 1 ? _in_complex(p) : _out_complex(p))) ? n ÷ 2 : n)
end

_shape(p::FFTPlan, s::Tuple) = p.howmany == 1 ? s : (s..., p.howmany)

function _input(p::FFTPlan, x::AbstractArray)
    length(x) == dim(p, 1) ||
        throw(DimensionMismatch("FFT input must have $(dim(p, 1)) elements ($(_shape(p, input_size(p)))), got $(length(x))"))
    if _in_complex(p)
        eltype(x) <: Number || throw(ArgumentError("FFT input must be numeric"))
        return convert(Array{ComplexF64}, x)
    else
        eltype(x) <: Real || throw(ArgumentError(":r2c input must be real"))
        return convert(Array{Float64}, x)
    end
end

"""
    execute!(out, plan, x) -> out

Apply `plan` to `x`, writing into `out`: an `Array{ComplexF64}` (complex output) or `Array{Float64}` (`:c2r`) with
`dim(plan, 2)` elements (any shape). `x` may have any shape with `dim(plan, 1)` elements; see [`FFTPlan`](@ref) for
the layout.
"""
function execute!(out::Array{T}, p::FFTPlan, x::AbstractArray) where {T}
    To = _out_complex(p) ? ComplexF64 : Float64
    T === To || throw(ArgumentError("output must be an Array{$To} for a :$(p.type) plan"))
    length(out) == dim(p, 2) || throw(DimensionMismatch("FFT output must have $(dim(p, 2)) elements, got $(length(out))"))
    xin = _input(p, x)
    _ptr(p)  # throws if closed
    GC.@preserve xin out begin
        _check(ccall((:sctljl_fft_execute, libsctljl), Cint, (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
                     p, Ptr{Float64}(pointer(xin)), length(xin) * (sizeof(eltype(xin)) ÷ 8),
                     Ptr{Float64}(pointer(out)), length(out) * (sizeof(T) ÷ 8)))
    end
    return out
end

"""
    execute(plan, x) -> y

Apply `plan` to `x` (any shape with `dim(plan, 1)` elements, e.g. `(input_size(plan)..., howmany)`). Returns an
array of size `output_size(plan)` when `howmany == 1`, else `(output_size(plan)..., howmany)`; its element type is
`ComplexF64`, or `Float64` for `:c2r`.
"""
function execute(p::FFTPlan, x::AbstractArray)
    out = Array{_out_complex(p) ? ComplexF64 : Float64}(undef, _shape(p, output_size(p)))
    return execute!(out, p, x)
end
