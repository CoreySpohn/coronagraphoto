# Rate vs readout

coronagraphoto exposes two parallel families of simulation functions
for every source type (`star`, `planet`, `disk`, `zodi`, and the
whole-scene aggregator `system`):

| Family | Returns | Stochastic? | Differentiable? |
|---|---|---|---|
| `*_rate` | Per-pixel count rate (e/s/pixel) | No | Yes (through the whole forward model) |
| `*_readout` | Per-pixel electron count (e) for one exposure | Yes (Poisson + binomial QE) | No |

This page explains when to reach for which.

## The mathematics

For a single source $S$ through the optical path $\mathcal{O}$, the
per-pixel **expected electron count** during an exposure of length
$t$ on a detector with quantum efficiency $\eta$ is:

$$
\mathbb{E}[N_{\text{electrons}}] \;=\; r_S(\mathcal{O}) \cdot t \cdot \eta
$$

where $r_S(\mathcal{O})$ is the count rate (electrons per second per
pixel). The `*_rate` functions return $r_S(\mathcal{O})$.

The actual observed electron count is a Poisson realisation of that
expectation. The `*_readout` functions return:

$$
N_{\text{electrons}} \;\sim\; \mathrm{Poisson}(r_S(\mathcal{O}) \cdot t \cdot \eta)
$$

For the high-count regime coronagraphoto also offers a "thinned" variant
that uses the Poisson thinning theorem
($\mathrm{Binomial}(\mathrm{Poisson}(\lambda), p) \sim \mathrm{Poisson}(\lambda p)$)
to fuse the QE binomial draw into the Poisson draw, which is faster
without changing the distribution. That's the `readout_source_electrons_thinned`
path inside the detector model.

## Use `*_rate` for fitting

Anything that needs gradients should differentiate through the rate
pipeline. Examples:

- **Likelihood evaluation** -- the rate map IS the model expectation;
  compare against observed counts via a Poisson or Gaussian likelihood.
- **MAP / HMC retrieval** -- gradient-based parameter inference (planet
  position, atmospheric composition, orbital elements).
- **Sensitivity studies** -- $\partial(\text{observable}) / \partial(\text{parameter})$
  for, e.g., the dependence of detection significance on telescope
  diameter or exposure time.
- **Yield calculator forward model** -- the deterministic
  rate-summed-over-targets that downstream optimisers query.

```python
import jax
import equinox as eqx
from coronagraphoto import system_rate

# Differentiable rate pipeline -- gradient flows end-to-end
@eqx.filter_jit
def total_electrons(wavelength_nm, scene, optical_path):
    rate_map = system_rate(
        scene, optical_path,
        start_time_jd=2_460_000.0,
        wavelength_nm=wavelength_nm,
        bin_width_nm=50.0,
        telescope_pa_deg=0.0,
        ecliptic_lat_deg=0.0, solar_lon_deg=135.0,
    )
    return rate_map.sum() * EXPOSURE_S

grad_fn = eqx.filter_grad(total_electrons)
```

Differentiable parameters can be any JAX array inside `scene` or
`optical_path` -- use `eqx.tree_at` to swap a field, or differentiate
directly w.r.t. a kwarg like `wavelength_nm`. The Richardson-converged
gradient tests in the test suite cover the wavelength case end-to-end.

The `system_rate` orchestrator sums all per-source rates -- star,
every planet, disk if present, zodi if present -- into one differentiable
map for the entire scene.

## Use `*_readout` for data generation

Anything that needs a noisy detector image should use the readout
family. Examples:

- **Generating an HWO survey dataset** -- thousands of frames, each
  with independent Poisson noise.
- **End-to-end ADI / RDI demos** -- generate noisy science frames at
  varying telescope rotations, then run KLIP on the cube.
- **Comparing post-processing algorithms** -- the algorithm should see
  realistic noise, not the noiseless rate.

```python
from coronagraphoto import system_readout
import jax

key = jax.random.PRNGKey(0)
image = system_readout(
    scene, optical_path, key,
    start_time_jd=..., exposure_time_s=..., wavelength_nm=...,
    bin_width_nm=..., telescope_pa_deg=...,
    ecliptic_lat_deg=..., solar_lon_deg=...,
)
```

`system_readout` independently draws Poisson noise per source (each
gets its own PRNG subkey) and sums the realisations. The sum of
Poissons is Poisson with summed mean, so this is statistically
equivalent to one combined draw, but the per-source split gives you
deterministic noise for the same key + scene combination.

## What you cannot do

You cannot differentiate through `*_readout`. The Poisson draw is a
discrete random integer and its gradient is zero almost everywhere.
JAX will not raise; it will silently return zero gradients. If you find
yourself doing `jax.grad(simulate_frame)` where `simulate_frame` calls
`*_readout`, **the result is meaningless**. Two acceptable workarounds:

1. **Differentiate the rate instead.** The expected count
   $r_S \cdot t \cdot \eta$ is what almost every inference problem
   actually wants, and it's differentiable.
2. **Use a Gaussian approximation** at high counts:
   $\mathrm{Poisson}(\lambda) \approx \mathcal{N}(\lambda, \sqrt{\lambda})$.
   The Gaussian distribution is reparameterisable and differentiable
   through the noise sample. Build it yourself if you need it.

## Performance

Both families share most of the work -- the readout family is just the
rate family followed by a single Poisson draw via the detector's
readout method. The Poisson draw is a few percent of the per-frame
cost on GPU. See [performance](performance) for the closure-vs-args
trap that dominates compile time and how to avoid it.
