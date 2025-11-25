# Functional Simulation Approach

`coronagraphoto` adopts a functional programming paradigm powered by JAX. This ensures that simulations are deterministic, easy to parallelize, and compatible with JIT compilation.

The simulation pipeline is built on three hierarchical levels: **Observation**, **Exposure**, and **Band**.

## 1. The Hierarchy

### Observation (`sim_observation`)
An **Observation** represents a sequence of data collection events over time. It is essentially a collection of *Exposures*.
*   **Goal**: Generate a time-series of images (e.g., for orbit characterization).
*   **Implementation**: Typically implemented by mapping (`jax.vmap` or Python loop) the `Exposure` simulation over a vector of times.

### Exposure (`sim_exposure`)
An **Exposure** represents a single readout of the detector. It accounts for the integration of photons over a specific duration and spectral bandwidth.
*   **Goal**: Produce a single 2D image of photoelectron counts (including noise).
*   **Process**:
    1.  Simulate the flux on the detector for multiple spectral bins (wavelenths).
    2.  Sum the electrons from all spectral bins.
    3.  Add detector noise (Read noise, Dark current, etc.).
*   **Code Reference**: Composed using `jax.vmap` over `sim_band`.

### Band (`sim_band`)
A **Band** represents the physics of light propagation for a single spectral element (wavelength+bin width).
*   **Goal**: Calculate the instantaneous photon rate on the detector for a specific wavelength.
*   **Process**:
    1.  Calculate source positions (Star, Planets) at the specific time.
    2.  Calculate source fluxes at the specific wavelength.
    3.  Propagate light through the `OpticalPath` (Primary -> Coronagraph -> Detector).
    4.  Apply PSF convolution and transmission losses.
*   **Code Reference**: `coronagraphoto.core.simulation.sim_star`, `sim_planets`, `sim_disk`, `sim_zodi`.

## 2. JAX Implementation Details

### The `Exposure` Object
The `Exposure` class is a JAX-compatible PyTree (via `Equinox`) that holds all parameters required to define a single integration:
*   `start_time_jd`: Scalar or Vector (for observations)
*   `exposure_time_s`: Duration
*   `central_wavelength_nm`: Vector (defining the spectral bins)
*   `bin_width_nm`: Vector (width of each bin)

### Vectorization (vmap)
We leverage `jax.vmap` to efficiently handle simulations over multiple spectral bands. Instead of looping over wavelengths in Python, we vectorize the physics logic over the `central_wavelength_nm` axis of the `Exposure` object. Typically this is followed by integrating the bands together to create the final image.

```python
# Pseudo-code for sim_exposure logic
def sim_exposure(exposure, optical_path, scene, key):
    # Vectorize physics over the wavelength axis (axis 0)
    # Exposure.in_axes helps specify which fields vary
    calc_spectral_electrons = jax.vmap(
        sim_band,
        in_axes=(Exposure.in_axes(central_wavelength_nm=0, bin_width_nm=0), ...)
    )

    # Get image for each wavelength in parallel
    spectral_images = calc_spectral_electrons(exposure, ...)

    # Sum to get broadband image
    total_image = jnp.sum(spectral_images, axis=0)

    # Add noise
    return total_image + detector_noise
```

### Optimization
Because the system is stateless and functional:
*   **Static Rates**: For static objects (stars, disks, local zodi) or when geometry doesn't change significantly, photon rates can be pre-calculated and cached outside the main time-loop.
*   **JIT**: The entire pipeline can be Just-In-Time compiled to XLA using Equinox's `filter_jit`, fusing operations for maximum speed and optionally for running on GPUs or TPUs.
