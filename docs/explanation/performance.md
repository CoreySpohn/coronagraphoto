# Performance: avoid baking large arrays into JIT compilation

When you wrap a simulation function with `@eqx.filter_jit`, any JAX array
referenced through a Python closure becomes a **constant baked into the
compiled program**, not a runtime argument. For a typical coronagraphoto
setup the PSF datacube held inside `optical_path.coronagraph` is the
dominant cost --- a 256x256 quarter-symmetric cube is 4.36 GB in float32
--- and JAX will print a warning at lowering time:

```
UserWarning: A large amount of constants were captured during lowering
(4.46GB total). If this is intentional, disable this warning by setting
JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1.
```

The visible symptoms are a long first-call compile (we measured 43.7 s
for an uncached forward model on a single-GPU host), the warning above,
and brittle reuse: any change to `optical_path` forces a full recompile
because the cube has been folded into the program's constant pool.

## The rule: hoist objects that carry JAX arrays into the signature

`eqx.filter_jit` traces JAX arrays in pytree **arguments** as runtime
inputs. Move every container that owns a heavy array --- `optical_path`,
`star`, `planet`, `disk`, `zodi` --- out of closure capture and into the
function signature. The PSF datacube then flows in as a normal input
array rather than being memcopied into the compiled binary.

```python
# Avoid: optical_path closed over from outer scope.
@eqx.filter_jit
def simulate_frame(mjd, wavelength_nm, key):
    rate = planet_rate(planet, optical_path, ...)
    return optical_path.detector.readout_source_electrons(
        rate, EXPOSURE_S, key
    )
```

```python
# Prefer: every JAX-array-bearing object is an argument.
@eqx.filter_jit
def simulate_frame(optical_path, planet, mjd, wavelength_nm, key):
    rate = planet_rate(planet, optical_path, ...)
    return optical_path.detector.readout_source_electrons(
        rate, EXPOSURE_S, key
    )
```

## Measured impact

On a single-GPU benchmark of the full uncached forward model (star +
planet + disk + zodi + Poisson readout) the two patterns compile and
run as follows:

| Variant                  | Closure (baked) | As-argument |
| ------------------------ | --------------- | ----------- |
| First-call compile       | 43.7 s          | 5.6 s       |
| Steady-state median      | 22.7 ms         | 22.9 ms     |
| Steady-state std (n=20)  | 6.2 ms          | 0.3 ms      |
| Captured-constants size  | 4.46 GB         | none        |

Steady-state per-frame cost is unchanged --- closure capture does not
cost anything once the program is compiled --- but compile time drops
roughly 8x and the constants warning disappears. The variance
improvement is a secondary effect: the long compile spills into the
first timed iterations and inflates the standard deviation of the
closure-pattern measurements.

## When closure capture is fine

Small precomputed JAX arrays that exist specifically as cache state
should stay closed-over. The cached forward model in
`coronagraphoto`'s benchmarks pre-computes the star and disk count
rates once and folds them into the JIT --- each is a 256x256 float32
array (256 KB), and baking them is the entire point of the cached
variant. The rule applies to large containers like
`optical_path.coronagraph`, not to every closure-captured array.

## Diagnostic: find what JAX captured

To see exactly which Python frames produced the captured constants,
set the report environment variable before the first compile:

```python
import os
os.environ["JAX_CAPTURED_CONSTANTS_REPORT_FRAMES"] = "-1"
```

JAX will print the call sites that introduced each captured constant
to stderr. This is the fastest way to identify a closure-captured cube
that should have been an argument.
