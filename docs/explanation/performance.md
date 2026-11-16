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

Keep instrument geometry, bands and pointing concrete when compiling. Close
over the optical path, bands and pointing; pass scene leaves and time as
arguments to reuse the same compiled scene geometry:

```python notest
import equinox as eqx
from coronagraphoto import image_electrons


@eqx.filter_jit
def predict(system, time_d):
    return image_electrons(
        system,
        optical_path,
        bands,
        time_d=time_d,
        pointing=pointing,
        zodi=local_zodi,
    )


rates = predict(system, time_d)
rates.planet.block_until_ready()
```

Passing the entire optical path as a traced argument also traces the detector
pixel scale; a static grid cannot be constructed from that tracer. Geometry
and wavelength-band changes require a new trace. Scene contrast parameters
and time support JIT and gradients. Use float64 for faint-contrast calculations
and enable it before constructing the scene or tables.
