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

## Core Concepts

The simulation is structured around three hierarchical levels:

1.  **Observation**: A series of exposures over time (e.g., for orbit characterization).
2.  **Exposure**: A single integration reading the detector over a full bandpass.
3.  **Band**: The propagation of a single spectral bin (monochromatic).

## Design Philosophy: "Bring Your Own Physics"

You might notice that `coronagraphoto` does not provide a single, black-box `run_simulation()` function. This is intentional.

We provide **primitives** (e.g. functions like `sim_planets` or `sim_star` and objects to hold standard data like `OpticalPath` or `Exposure`), but we require you to **compose** them yourself. This ensures:

1.  **Transparency**: You know exactly what is in your image (e.g., did you include zodi? read noise? which sources are being simulated?).
2.  **Flexibility**: You can easily modify the pipeline (e.g., add a custom noise model, return spectral cubes instead of summed images, or simulate only specific sources).
3.  **Control**: You have full control over the simulation flow and can optimize it for your specific science case. You can define your observation function to roll between each exposure, or observe a second star for RDI and subtract the frames, or return each source separately. If you don't need any spectral data you can speed up your simulations by only calculating the exposure for the central wavelength with the full bandwidth.

The goal is to provide the building blocks, not a rigid pre-built structure, allowing you to construct exactly the images you need, at the fidelity you need.

## Quick Start

Here is a basic example of composing a simulation for a single exposure integrating 5 spectral bands:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from coronagraphoto import (
    Exposure, OpticalPath, load_sky_scene_from_exovista,
    conversions
)
from coronagraphoto.optical_elements import (
    PrimaryAperture, SimpleDetector, ConstantThroughputElement, from_yippy
)
from coronagraphoto.core.simulation import sim_star, sim_planets, sim_disk, sim_zodi
from yippy import Coronagraph as YippyCoronagraph

# 1. Define the Physics (Simulate all bands and sum)
def sim_band(exposure, optical_path, scene, key):
    """Simulate a single wavelength band."""
    # Split keys for different sources
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Unpack scalar params for this band
    args = (
        exposure.start_time_jd, exposure.exposure_time_s,
        exposure.central_wavelength_nm, exposure.bin_width_nm,
    )

    # Use the sim_* methods to calculate incident electrons from each
    # astrophysical source
    star_electrons = sim_star(*args, scene.stars, optical_path, k1)
    planet_electrons = sim_planets(*args, exposure.position_angle_deg, scene.planets, optical_path, k2)
    disk_electrons = sim_disk(*args, exposure.position_angle_deg, scene.disk, optical_path, k3)
    zodi_electrons = sim_zodi(*args, scene.zodi, optical_path, k4)

    return star_electrons + planet_electrons + disk_electrons + zodi_electrons

def sim_exposure(exposure, optical_path, scene, prng_key):
    """Simulate a single exposure/readout of the detector."""
    # Vectorize sim_band over the wavelength axis (axis 0)
    # We use Exposure.in_axes to specify which fields are vectors
    keys = jax.random.split(prng_key, exposure.central_wavelength_nm.shape[0])
    spectral_electrons = jax.vmap(
        sim_band,
        in_axes=(Exposure.in_axes(central_wavelength_nm=0, bin_width_nm=0), None, None, 0)
    )(exposure, optical_path, scene, keys)

    # Sum all spectral bins
    all_source_electrons = jnp.sum(spectral_electrons, axis=0)

    # Generate noise electrons
    noise_electrons = optical_path.detector.readout_noise_electrons(exposure.exposure_time_s, prng_key)

    return all_source_electrons + noise_electrons

# 2. Load the Scene (ExoVista) and Coronagraph (yippy)
scene = load_sky_scene_from_exovista("path/to/exovista_system.fits")
yippy_coro = YippyCoronagraph("path/to/coronagraph_data")
coronagraph = from_yippy(yippy_coro)

# 3. Define the Optical Path
optical_path = OpticalPath(
    primary=PrimaryAperture(diameter_m=6.0),
    attenuating_elements=(ConstantThroughputElement(throughput=0.9),),
    coronagraph=coronagraph,
    detector=SimpleDetector(pixel_scale=1/512, shape=(512, 512))
)

# 4. Define the Exposure data
exposure = Exposure(
    start_time_jd=conversions.decimal_year_to_jd(2025.0),
    exposure_time_s=3600.0,
    central_wavelength_nm=jnp.linspace(500, 600, 5), # 5 spectral bins
    bin_width_nm=jnp.full(5, 20.0),
    position_angle_deg=0.0
)

# 5. Compile and run simulation
key = jax.random.PRNGKey(0)
jit_sim_exposure = eqx.filter_jit(sim_exposure)
image = jit_sim_exposure(exposure, optical_path, scene, key)
```

For more advanced usage, including time-series animations, check the documentation (which I promise will exist eventually).
