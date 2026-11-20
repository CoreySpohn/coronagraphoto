# JAX and memory

`photon_rates` retains a `(W,Q,ny,nx)` image per source term. Prefer
`expected` for band-integrated output: its `lax.scan` carries only
`(W,ny,nx)` electron arrays, renders one node, applies detector yield with a
one-node `Quadrature`, and discards that node's images. `observe` uses this
streaming path. Reverse-mode differentiation can retain scan residuals.

## One program for any nodes and any roll

The wavelength nodes and weights, the roll and the epoch are traced array
leaves of the `Exposure`. The detector shape, grid axes, frame and
coronagraph band selection are static. Every node is rendered by one
`lax.scan` (or `lax.map` for `photon_rates`) whose body builds the resampling
operators from the traced node's lambda/D scale, so the program does not grow
with the node count and compiles once. A different node value, weight or roll
reuses it; a different node count, band count, detector shape or
instrument structure is a new array shape or static configuration and compiles
once more. There is no reader cache to size or clear: JAX's own
compilation cache holds the programs.

A coronagraph that dispatches by band (`MultiBandCoronagraph`) adds one
branch per model that the concrete nodes fall in, selected inside the program
from the traced node wavelength; traced nodes include every model.

Time, sky-background coordinates, array-valued scene fluxes, detector
yield/noise parameters, PRNG keys, the roll and the quadrature are runtime
inputs, and `jax.grad` differentiates through all of them. Use arrays for
numerical scene parameters that you intend to vary; Equinox treats Python
scalar leaves as static. A grid returned by `observe` records the roll only
when it is concrete.

The readers also compose with a caller's own JIT, vmap and grad. While tracing
they stage directly into that program. Pass the exposure, scene and instrument as
arguments, or close over them:

```python notest
from dataclasses import replace

import equinox as eqx
from coronagraphoto import build_exposure, observe

exposure = build_exposure(scene, instrument, bands, time_d=0.0, plan=plan)


@eqx.filter_jit
def predict(scene, instrument, key, time_d, roll_deg):
    at = replace(exposure, time_d=time_d, roll_deg=roll_deg)
    return observe(key, scene, instrument, at)


image = predict(scene, instrument, key, time_d, 30.0)
image.data.block_until_ready()
```

`vmap` adds a frame axis to interpolation and FFT workspaces. Its memory per
frame is implementation-dependent, not an inherent requirement of rendering
an observation. Prefer `lax.map` when frames would exhaust memory:

```python notest
import jax

images = jax.lax.map(
    lambda kt: predict(scene, instrument, kt[0], kt[1], 30.0).data,
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
A JIT function can hold more than one executable when device, sharding or JAX
compilation context changes. To release compiled programs between independent
experiments, call
[`jax.clear_caches()`](https://docs.jax.dev/en/latest/_autosummary/jax.clear_caches.html)
and release any explicitly retained compiled objects too. Cache release does
not promise an immediate drop in RSS: live outputs and allocator pools can
remain resident. It never changes numerical results.
