<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/coreyspohn/coronagraphoto/main/docs/_static/tmp_logo.png" alt="coronagraphoto logo" />
  <br><br>
</p>

<p align="center">
  <a href="https://pypi.org/project/coronagraphoto/"><img src="https://img.shields.io/pypi/v/coronagraphoto.svg?style=flat-square" alt="PyPI"/></a>
  <a href="https://coronagraphoto.readthedocs.io"><img src="https://readthedocs.org/projects/coronagraphoto/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/coreyspohn/coronagraphoto/blob/main/LICENSE"><img src="https://img.shields.io/github/license/coreyspohn/coronagraphoto?style=flat-square" alt="License"/></a>
  <a href="https://pypi.org/project/coronagraphoto/"><img src="https://img.shields.io/pypi/pyversions/coronagraphoto?style=flat-square" alt="Python"/></a>
  <a href="https://github.com/coreyspohn/coronagraphoto/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/coreyspohn/coronagraphoto/tests.yml?branch=main&logo=github&style=flat-square&label=tests" alt="Tests"/></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat-square&logo=pre-commit" alt="pre-commit"/></a>
</p>

---

# coronagraphoto

**coronagraphoto** is a Python library designed to simulate coronagraphic observations of exoplanetary systems. The base "thing" it produces are images/photos, hence the name. It has been designed to bridge the gap between yield calculations and concrete image generation for missions like the Habitable Worlds Observatory (HWO).

The library integrates high-fidelity coronagraph models from the standard format used for yield calculations (dubbed a "Yield Input Package" and loaded via **[yippy](https://github.com/CoreySpohn/yippy)**) with detailed planetary system simulations (via **[ExoVista](https://github.com/alexrhowe/ExoVista)**) to produce realistic detector images.

Built on **JAX**, `coronagraphoto` is fully JIT-compilable, differentiable, and GPU-accelerated, making it suitable for large-scale optimization and high-performance simulation.

## Key Features

*   **End-to-End Simulation**: From astrophysical scenes to detector readouts.
*   **JAX & JIT Compatible**: High-performance simulations using functional programming patterns.
*   **Modular Design**: flexible optical paths, easily swappable coronagraphs and detectors.
*   **HWO Ready**: Specifically designed to support yield modeling for future direct imaging missions.

## Installation

```bash
pip install coronagraphoto
```

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements (CUDA/TPU/CPU).)*

## Design philosophy: "Bring your own physics"

`coronagraphoto` does not provide a single, black-box `run_simulation()` function. It provides **primitives** (per-source `sim_*` functions, an `OpticalPath`, detectors and throughput elements) and a thin orchestrator (`sim_system`) that sums them. The convention:

- Per-source simulators: `sim_<source>(source, optical_path, prng_key, *, observation_kwargs)` -- one source, one detector readout.
- Whole-scene orchestrator: `sim_system(scene, optical_path, prng_key, *, observation_kwargs)` -- sums star + planets + disk + zodi from a `skyscapes.Scene`.

This keeps the pipeline transparent (you know exactly which sources contributed), flexible (drop in custom noise, return spectral cubes, RDI two scenes), and fast (each per-source kernel is JIT-cached at its natural shape boundary).

## Quick start

```python
import jax
from coronagraphoto import (
    OpticalPath, PrimaryAperture, IdealDetector,
    load_scene_from_exovista, sim_system,
)
from coronagraphoto.optical_elements import ConstantThroughput
from yippy import EqxCoronagraph

# 1. Load a skyscapes.Scene (system + default zodi) from ExoVista.
scene = load_scene_from_exovista("path/to/exovista_system.fits")

# 2. Build the optical path.
coronagraph = EqxCoronagraph("path/to/coronagraph_data")
optical_path = OpticalPath(
    primary=PrimaryAperture(diameter_m=6.0),
    attenuating_elements=(ConstantThroughput(throughput=0.9),),
    coronagraph=coronagraph,
    detector=IdealDetector(pixel_scale_arcsec=0.01, shape=(512, 512)),
)

# 3. Simulate one detector readout.
image = sim_system(
    scene,
    optical_path,
    jax.random.PRNGKey(0),
    start_time_jd=2_460_000.0,
    exposure_time_s=3600.0,
    wavelength_nm=550.0,
    bin_width_nm=50.0,
    telescope_pa_deg=0.0,
    ecliptic_lat_deg=0.0,
    solar_lon_deg=135.0,
)
```

For broadband / IFS simulations, `jax.vmap` over `wavelength_nm` (and sum or stack the result) -- the kwarg-only signature is designed so the wavelength axis is a clean vmap target.
