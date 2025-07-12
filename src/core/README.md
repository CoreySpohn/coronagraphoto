# Coronagraphoto v2: Observatory & Reduction Framework

A medium-fidelity coronagraph observation simulation tool built around a functional approach for light path simulation.

## Overview

This is a complete rewrite of the coronagraphoto simulation framework, implementing the architecture described in `docs/refactor_architecture.md`. The new v2 architecture provides:

- **Pure functional light paths** - Easy to test, optimize, and reason about
- **Flexible observation planning** - Builder patterns for complex observing sequences
- **Modular post-processing** - Composable reduction pipelines
- **Clear separation of concerns** - Planning, execution, and reduction are distinct phases
- **JAX-ready design** - Prepared for high-performance acceleration

## Architecture Components

### Core Data Structures

- **`Target`** - Wrapper around scene files (ExoVista, synthetic scenes)
- **`Observation`** - Immutable specification of a single exposure
- **`ObservationSequence`** - Ordered list of observations with builder methods
- **`IntermediateData`** - Multi-component container preserving source separation
- **`PropagationContext`** - Time/wavelength state for light propagation

### Execution Engine

- **`Observatory`** - Pure executor that runs sequences through light paths
- **`LightPath`** - List of pure functions representing the optical system
- **Light Path Functions** - Stateless physics implementations (primary, coronagraph, detector, etc.)

### Post-Processing

- **`ReductionPipeline`** - Sequential processing of data products
- **`ReductionStep`** - Individual algorithms (RDI, ADI, stacking, etc.)

## Quick Start

```python
from functools import partial
from astropy.time import Time
import astropy.units as u

from coronagraphoto.observation import Target, ObservationSequence
from coronagraphoto.observatory import Observatory
from coronagraphoto.light_paths import (
    apply_primary, apply_coronagraph, apply_detector,
    PrimaryParams, CoronagraphParams, DetectorParams
)
from coronagraphoto.reduction import ReductionPipeline, Derotate, StackFrames

# 1. Define hardware
primary = PrimaryParams(diameter=8*u.m, reflectivity=0.95)
coronagraph = CoronagraphParams(
    inner_working_angle=0.1*u.arcsec,
    outer_working_angle=1.0*u.arcsec,
    contrast=1e-10
)
detector = DetectorParams(
    pixel_scale=0.02*u.arcsec,
    read_noise=3*u.electron,
    dark_current=0.01*u.electron/u.s
)

# 2. Create light paths
science_path = [
    partial(apply_primary, params=primary),
    partial(apply_coronagraph, params=coronagraph),
    partial(apply_detector, params=detector),
]

# 3. Configure observatory
observatory = Observatory({"science": science_path})

# 4. Plan observations
target = Target("scene.fits", name="My Target")
sequence = ObservationSequence.for_adi(
    target=target,
    path_name="science",
    n_exposures=8,
    exposure_time=300*u.s,
    start_time=Time("2024-01-01T00:00:00")
)

# 5. Execute
raw_data = observatory.run(sequence, seed=42)

# 6. Process
pipeline = ReductionPipeline([
    Derotate(),
    StackFrames(method="mean")
])
final_data = pipeline.process(raw_data)
```

## Directory Structure

```
src/core/coronagraphoto/
├── __init__.py              # Package initialization
├── observation.py           # Target, Observation, ObservationSequence
├── data_models.py           # IntermediateData, PropagationContext
├── observatory.py           # Observatory execution engine
├── light_paths.py           # Physics functions and parameter classes
├── reduction.py             # Post-processing pipeline
├── example_usage.py         # Complete demonstration
└── tests/
    ├── test_observation.py   # Unit tests for observation components
    └── test_integration.py   # End-to-end integration tests
```

## Key Features

### 1. Pure Functional Light Paths

Light paths are sequences of pure functions that transform data:

```python
def apply_primary(data: IntermediateData, params: PrimaryParams, context: PropagationContext) -> IntermediateData:
    """Apply primary mirror effects - collecting area, reflectivity."""
    collecting_area = np.pi * (params.diameter / 2) ** 2
    star_flux = data.star_flux * params.reflectivity * collecting_area
    return data.update(star_flux=star_flux)
```

### 2. Flexible Observation Planning

Builder patterns for common observing modes:

```python
# Angular Differential Imaging
adi_seq = ObservationSequence.for_adi(
    target=target,
    path_name="science",
    n_exposures=10,
    exposure_time=300*u.s,
    start_time=start_time,
    total_roll_angle=360*u.deg
)

# Reference Differential Imaging
rdi_seq = ObservationSequence.for_rdi(
    science_target=science_target,
    ref_target=ref_target,
    science_path_name="science",
    ref_path_name="reference",
    exposure_time=300*u.s,
    start_time=start_time
)
```

### 3. Modular Post-Processing

Composable reduction steps:

```python
pipeline = ReductionPipeline([
    ReferenceSubtract(method="normalized_subtraction"),
    Derotate(reference_angle=0*u.deg),
    StackFrames(method="mean"),
    BackgroundSubtract(method="annulus"),
    ContrastCurve(separations=np.linspace(0.1, 1.0, 10))
])
```

## Implementation Status

### ✅ Completed (Phase 1 & 2)

- **Foundational Components**: Target, Observation, ObservationSequence with builder patterns
- **Core Execution Engine**: Observatory with functional light path execution
- **Data Models**: IntermediateData and PropagationContext
- **Basic Physics**: Primary, coronagraph, filter, detector models
- **Post-Processing Framework**: ReductionPipeline with common algorithms

### 🔄 In Progress (Phase 3)

- **Advanced Light Path Functions**: More sophisticated physics models
- **Scene Loading**: ExoVista and synthetic scene generators
- **Comprehensive Testing**: Property-based tests with hypothesis

### 📋 Future (Phase 4-6)

- **Performance Optimization**: JAX integration and JIT compilation
- **Advanced Capabilities**: Time-variable effects, fully optimized pipelines
- **Integration**: Connection to existing coronagraphoto ecosystem

## Testing

Run the integration tests to verify functionality:

```bash
cd src/core
python tests/test_integration.py
```

Run the demonstration:

```bash
cd src/core
python example_usage.py
```

## Architecture Advantages

1. **Testability**: Pure functions are easy to unit test
2. **Composability**: Light paths can be easily reconfigured
3. **Performance**: Ready for JAX acceleration
4. **Clarity**: Clear separation between planning, execution, and reduction
5. **Extensibility**: Easy to add new physics models or reduction steps

## Relationship to Original Codebase

This v2 architecture is designed to **complement** the existing coronagraphoto codebase in `src/coronagraphoto/`. Key differences:

- **Functional vs Object-Oriented**: v2 uses pure functions, v1 uses classes
- **Explicit vs Implicit**: v2 makes all parameters explicit, v1 uses configuration files
- **Modular vs Monolithic**: v2 separates concerns, v1 combines them in sessions
- **Testable vs Complex**: v2 prioritizes testability, v1 prioritizes feature completeness

## Development Philosophy

Following the principles outlined in `docs/development_strategy.md`:

1. **Test-Driven Development**: Write tests first, then implementation
2. **Clarity Before Optimization**: Focus on correctness and readability
3. **Incremental Implementation**: Build in phases with working prototypes
4. **Pure Functions**: Stateless, side-effect-free computation

## Future JAX Integration

The architecture is designed for seamless JAX integration:

```python
# Future Phase 5 - JAX-accelerated light paths
import jax.numpy as jnp
from jax import jit, vmap

@jit
def jax_light_path(data: jnp.ndarray, params: dict) -> jnp.ndarray:
    """JIT-compiled entire light path for maximum performance."""
    # Composed JAX-native functions
    return data  # placeholder

# Observatory can automatically detect and use JAX paths
observatory = Observatory({"fast_path": jax_light_path})
```

This design ensures a smooth transition from the current Python implementation to high-performance JAX execution while maintaining the same user interface.