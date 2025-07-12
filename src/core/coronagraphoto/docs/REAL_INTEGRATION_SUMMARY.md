# Real Integration Summary: Coronagraphoto v2 Light Path Patterns

## Overview

The new coronagraphoto v2 light path patterns have been successfully updated to integrate with **real yippy coronagraph models** and **ExoVista scene files**. This addresses your concern about the patterns working with actual components rather than just placeholders.

## Integration Achieved

### ✅ Real Yippy Coronagraph Integration

The new patterns now properly load and use yippy `Coronagraph` objects:

```python
# Create coronagraph parameters pointing to real model directory
coronagraph_params = CoronagraphParams(
    inner_working_angle=0.1 * u.arcsec,
    outer_working_angle=1.0 * u.arcsec,
    coronagraph_dir="input/coronagraphs/demo",  # Points to real yippy model
    use_jax=True,
    cpu_cores=4
)

# The pattern automatically loads the yippy Coronagraph object
path = PathPipeline(Path.primary(primary_params)) >> Path.coronagraph(coronagraph_params)
```

**Key improvements:**
- `CoronagraphParams.get_coronagraph_model()` automatically loads yippy models
- `apply_star_coronagraph()` uses real `coronagraph.stellar_intens()` when available
- Fallback to simple suppression if yippy not available
- Full integration with yippy JAX support

### ✅ Real ExoVista Scene Integration

The new patterns now properly load ExoVista scenes:

```python
# Create target from real ExoVista scene file
target = Target("input/scenes/more_pix.fits", name="more_pix_system")

# Scene is automatically loaded with full ExoVista integration
scene_data = target.load_scene()
```

**Key improvements:**
- `load_scene_from_exovista()` extracts star, planet, and disk components
- Proper handling of ExoVista flux density calls
- Multi-component scene data preserved in `IntermediateData`
- Graceful fallback if ExoVista not available

### ✅ Complete End-to-End Integration

All four light path patterns work identically with real components:

```python
# All patterns use the SAME real yippy coronagraph model
pipeline_path = (PathPipeline(Path.primary(primary_params)) >> 
                 Path.coronagraph(coro_params) >>  # Real yippy model
                 Path.detector(det_params))

functional_path = compose(
    Path.detector(det_params),
    Path.coronagraph(coro_params),  # Same real yippy model
    Path.primary(primary_params)
)

# Both work identically with real ExoVista scenes
result = pipeline_path(exovista_data, context)
```

## File Structure Integration

### Coronagraph Models
```
input/coronagraphs/
├── demo/                    # Your demo coronagraph
│   ├── psf_datacube.fits   # yippy PSF data
│   ├── config.yaml         # yippy configuration
│   └── ...                 # Other yippy files
└── [your_models]/          # Add your own models here
```

### Scene Files
```
input/scenes/
├── more_pix.fits           # Your ExoVista scene
└── [your_scenes]/          # Add your own scenes here
```

## Usage Examples

### Basic Real Usage

```python
from coronagraphoto.light_paths import (
    Path, PathPipeline, CoronagraphParams, PrimaryParams, DetectorParams
)
from coronagraphoto.observation import Target, ObservationSequence
from coronagraphoto.observatory import Observatory

# 1. Real hardware with yippy coronagraph
primary_params = PrimaryParams(diameter=6.5 * u.m, reflectivity=0.95)
coro_params = CoronagraphParams(
    inner_working_angle=0.1 * u.arcsec,
    outer_working_angle=1.0 * u.arcsec,
    coronagraph_dir="input/coronagraphs/demo"  # Real yippy model
)
detector_params = DetectorParams(pixel_scale=0.01 * u.arcsec, ...)

# 2. Real light path using Pipeline pattern
science_path = (PathPipeline(Path.primary(primary_params)) >> 
               Path.coronagraph(coro_params) >>  # Uses real yippy
               Path.detector(detector_params))

# 3. Real ExoVista target
target = Target("input/scenes/more_pix.fits")  # Real ExoVista scene

# 4. Real observation sequence
obs_seq = ObservationSequence.for_adi(
    target=target,
    path_name="science_imaging",
    n_exposures=8,
    exposure_time=300 * u.s,
    start_time=Time.now()
)

# 5. Real simulation execution
observatory = Observatory({"science_imaging": science_path})
result = observatory.run(obs_seq)  # Uses real yippy + ExoVista
```

### Advanced Real Integration

```python
# Multiple real coronagraphs
light_paths = {
    "vortex_coronagraph": PathPipeline(...) >> Path.coronagraph(
        CoronagraphParams(coronagraph_dir="input/coronagraphs/vortex")
    ),
    "lyot_coronagraph": PathPipeline(...) >> Path.coronagraph(
        CoronagraphParams(coronagraph_dir="input/coronagraphs/lyot")
    ),
    "reference_imaging": PathPipeline(...) >> Path.detector(...)  # No coronagraph
}

# RDI with different coronagraphs
rdi_sequence = ObservationSequence.for_rdi(
    science_target=Target("input/scenes/science_target.fits"),
    ref_target=Target("input/scenes/reference_star.fits"),
    science_path_name="vortex_coronagraph",  # Real vortex coronagraph
    ref_path_name="reference_imaging",       # No coronagraph
    exposure_time=1800 * u.s,
    start_time=Time.now()
)
```

## Key Technical Details

### Coronagraph Integration

1. **Loading**: `CoronagraphParams.get_coronagraph_model()` loads yippy objects
2. **Physics**: `apply_star_coronagraph()` calls `coronagraph.stellar_intens()`
3. **Configuration**: Full support for yippy parameters (JAX, CPU cores)
4. **Fallback**: Graceful degradation if yippy not installed

### ExoVista Integration

1. **Loading**: `load_scene_from_exovista()` uses `ExovistaSystem(scene_path)`
2. **Components**: Extracts star, planets, disk into `IntermediateData`
3. **Physics**: Calls `spec_flux_density()` for realistic fluxes
4. **Fallback**: Graceful degradation if exoverses not installed

### Pattern Compatibility

All four patterns work identically:
- **Pipeline**: `Path.primary() >> Path.coronagraph() >> Path.detector()`
- **Functional**: `compose(detector, coronagraph, primary)`
- **Builder**: `PathBuilder().primary().coronagraph().detector().build()`
- **Registry**: `Path.from_registry("coronagraph", params)`

## Benefits Over Old Pattern

### Before (Cumbersome)
```python
# Had to manually integrate with yippy and ExoVista
coro = Coronagraph("input/coronagraphs/demo", use_jax=True)
system = ExovistaSystem("input/scenes/more_pix.fits")
obs = Observation(coro, system, ...)  # Tightly coupled
```

### After (Elegant)
```python
# Clean separation and composable patterns
coro_params = CoronagraphParams(coronagraph_dir="input/coronagraphs/demo")
target = Target("input/scenes/more_pix.fits")
path = PathPipeline(Path.coronagraph(coro_params))  # Loosely coupled
```

**Improvements:**
- ✅ **Separation of concerns**: Hardware config separate from observation planning
- ✅ **Reusable components**: Same coronagraph used in multiple paths
- ✅ **Pattern consistency**: All patterns work with real components
- ✅ **Graceful fallbacks**: Works even if dependencies missing
- ✅ **Type safety**: Full type annotations for IDE support

## Installation Requirements

For full functionality, install optional dependencies:

```bash
# For yippy coronagraph models
pip install yippy

# For ExoVista scene files
pip install exoverses

# Both are optional - patterns work with fallbacks if missing
```

## Next Steps

1. **Use real components**: Point to `input/coronagraphs/demo` and `input/scenes/more_pix.fits`
2. **Add your models**: Place your yippy coronagraphs in `input/coronagraphs/`
3. **Add your scenes**: Place your ExoVista scenes in `input/scenes/`
4. **Start with Pipeline**: Use the Pipeline pattern for most use cases
5. **Explore patterns**: Try Builder pattern for complex conditional logic

## Summary

The new light path patterns now provide **complete integration** with real yippy coronagraph models and ExoVista scene files, addressing your original concern about working with actual components rather than placeholders. The patterns are much cleaner than the old manual approach while maintaining full compatibility with the existing physics models.

**The cumbersome pattern has been replaced with elegant functional composition that works with real data!** 🎉