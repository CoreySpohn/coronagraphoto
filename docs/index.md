# coronagraphoto

A JAX-accelerated coronagraphic observation simulator for HWO mission planning.

```{toctree}
:maxdepth: 2
:caption: Documentation

simulating_zodi_with_telescope_orbit
```

```{toctree}
:maxdepth: 2
:caption: API Reference

autoapi/coronagraphoto/index
```

## Overview

**coronagraphoto** is a Python library for simulating coronagraphic observations of exoplanetary systems. It provides:

- JAX-native implementations for GPU/TPU acceleration
- Modular source models (stars, planets, disks, zodiacal light)
- Coronagraph integration via [yippy](https://github.com/CoreySpohn/yippy)
- Realistic detector noise models
- Integration with [coronalyze](https://github.com/CoreySpohn/coronalyze) for post-processing

## Installation

```bash
pip install coronagraphoto
```

## Quick start

See the [README](https://github.com/CoreySpohn/coronagraphoto#quick-start) for a full end-to-end example. In short:

```python
import jax
from coronagraphoto import (
    OpticalPath, PrimaryAperture, SimpleDetector,
    load_scene_from_exovista, system_readout,
)
from coronagraphoto.optical_elements import ConstantThroughputElement
from yippy import EqxCoronagraph

scene = load_scene_from_exovista("path/to/exovista_system.fits")
optical_path = OpticalPath(
    primary=PrimaryAperture(diameter_m=6.0),
    attenuating_elements=(ConstantThroughputElement(throughput=0.9),),
    coronagraph=EqxCoronagraph("path/to/coronagraph_data"),
    detector=SimpleDetector(pixel_scale=0.01, shape=(512, 512)),
)
image = system_readout(
    scene, optical_path, jax.random.PRNGKey(0),
    start_time_jd=2_460_000.0, exposure_time_s=3600.0,
    wavelength_nm=550.0, bin_width_nm=50.0,
    telescope_pa_deg=0.0,
    ecliptic_lat_deg=0.0, solar_lon_deg=135.0,
)
```
