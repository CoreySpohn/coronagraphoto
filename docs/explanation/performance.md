# JAX and memory

`image_rates` retains a `(W,Q,ny,nx)` image per source term. Prefer
`image_electrons` for band-integrated output: its `lax.scan` carries only
`(W,ny,nx)` electron arrays, renders one node, applies detector yield with a
one-node `Quadrature`, and discards that node's images. `observe` uses this
streaming path. Reverse-mode differentiation can retain scan residuals.

`PixelGrid` and `Frame` are immutable host geometry. The reader prepares
static node branches selected by the scan, so compile size grows with node
count. It does not precompute or retain all node images. The table adapter's
reference sampling is scaled to each concrete node using shared grid tools.

All three public readers share a cache of at most four eager JIT functions.
The first eager call compiles; later calls reuse that kernel when static
configuration and array shapes/dtypes (including weak scalar types) match.
The cache evicts the least recently used kernel, so returning to an evicted
configuration compiles again. `observe` shares the `image_electrons` renderer,
then draws detector noise in a separate small cached kernel; it does not
retain a second image-rendering kernel. Rendering and noise share the same
four-entry limit.

Time, sky-background coordinates, array-valued scene fluxes, detector yield/noise parameters and PRNG keys remain runtime
inputs. Use arrays for numerical scene parameters that you intend to vary;
Equinox treats Python scalar leaves as static. Bands, pointing, quadrature
order, detector pixel scale/shape and coronagraph grids are static geometry.
Changing these requires a new compilation. Numerical tables remain array
inputs wherever the instrument exposes them as PyTree leaves.

A supplied `Quadrature` is encoded by shape, dtype and exact node/weight bytes
into an immutable, hashable kernel key. Equal rules share a kernel even when
they are different objects; changing either array selects a new kernel. This
small host copy is intentional: each wavelength requires concrete resampling
geometry. Build the quadrature before tracing and close over it in an outer
JIT; passing its arrays as traced arguments is unsupported. The default rule
is built once while tracing the kernel. A supplied rule supersedes the order.

The readers also compose with a caller's own JIT. While tracing they stage
directly into that program, without adding an eager cache entry. Pass the
optical path as an argument, or close over it. Its detector pixel scale is a
validated host float and stays static in either form:

```python notest
import equinox as eqx
from coronagraphoto import observe


@eqx.filter_jit
def predict(system, optical_path, key, time_d):
    return observe(
        key,
        system,
        optical_path,
        bands,
        plan,
        time_d=time_d,
        pointing=pointing,
        zodi=local_zodi,
    )


image = predict(system, optical_path, key, time_d)
image.data.block_until_ready()
```

`vmap` adds a frame axis to interpolation and FFT workspaces. Its memory per
frame is implementation-dependent, not an inherent requirement of rendering
an observation. Prefer `lax.map` when frames would exhaust memory:

```python notest
import jax

images = jax.lax.map(
    lambda kt: predict(system, optical_path, kt[0], kt[1]).data,
    (keys, times_d),
)
images.block_until_ready()
```

This retains the output stack while reusing one frame's workspace. Compilation
and table loading also consume memory beyond the executable's temporary
buffers. Scene contrast parameters and time support gradients through the
deterministic readers. Use float64 for faint-contrast calculations and enable
it before constructing the scene or tables.

On the tutorial table, the largest workspaces are padded PSF FFTs over the
four nearest offsets for every planet. For eight planets and 256-square
PSFs, two complex128 buffers of shape `(8, 4, 1024, 513)` occupy about
256.5 MiB each. The complete offset table is not FFT-shifted for each query.
Streaming individual planets can change floating-point rounding, so the
reader retains batched planet interpolation. Dense resampling matrices and
node image accumulators are much smaller than those FFT workspaces.

Executables, staging graphs, constants and allocator pools contribute to
resident memory in addition to the buffers reported by `memory_analysis()`.
The eager cache retains up to four configurations across all three readers;
new reader kinds, pointings, bands or shapes can consume additional entries.
A JIT function can hold more than one executable when device, sharding or JAX
compilation context changes. The four-entry limit bounds functions rather
than bytes or all possible backend executable variants.

To release these entries between independent notebook sections:

```python notest
from coronagraphoto import clear_reader_cache

clear_reader_cache()
```

This does not clear caller-owned JIT kernels. Reuse the same compiled
function across frames; constructing new JIT closures or retaining objects
returned by `lower(...).compile()` keeps additional programs alive. To clear
JAX's compilation and staging caches between independent experiments, call
[`jax.clear_caches()`](https://docs.jax.dev/en/latest/_autosummary/jax.clear_caches.html).
Release any explicitly retained compiled objects too. Cache release does
not promise an immediate drop in RSS: live outputs and allocator pools can
remain resident. Neither cache-clear operation changes numerical results.
