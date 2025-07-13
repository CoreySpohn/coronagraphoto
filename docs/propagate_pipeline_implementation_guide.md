# Pipeline Implementation Guide: Propagate Method Architecture

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Propagate Method Pattern](#propagate-method-pattern)
4. [Source-Specific Pipelines](#source-specific-pipelines)
5. [Implementation Details](#implementation-details)
6. [JAX Optimization](#jax-optimization)
7. [Usage Examples](#usage-examples)
8. [Testing Strategy](#testing-strategy)

## Overview

This implementation uses a `.propagate()` method chain wrapped in a `Pipeline` class, with separate pipelines for each source type (star, planets, disk, etc.). This approach:

- **Avoids conditionals** for better JAX performance
- **Separates concerns** by source type
- **Maintains type safety** with consistent array shapes
- **Enables parallel execution** of different source pipelines

## Core Architecture

### Basic Pattern
```python
# Create a pipeline for star source
star_pipeline = Pipeline(
    primary.propagate(coronagraph).propagate(filter).propagate(detector)
)

# Execute
star_image = star_pipeline(star_source)
```

### Key Principles

1. **Each component has a `.propagate()` method** that returns a new component
2. **Pipeline wraps the final chain** and handles execution
3. **Separate pipelines for each source** (star, planets, disk)
4. **Return zeros for unused sources** to maintain consistent shapes

## Propagate Method Pattern

### Base Component Class

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np
import jax.numpy as jnp

T = TypeVar('T', bound='OpticalComponent')

class OpticalComponent(ABC, Generic[T]):
    """Base class for all optical components."""
    
    @abstractmethod
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply the component's effect to the field."""
        pass
    
    def propagate(self: T, next_component: 'OpticalComponent') -> 'ChainedComponent[T]':
        """Chain this component with the next one."""
        return ChainedComponent(self, next_component)

class ChainedComponent(OpticalComponent):
    """Represents a chain of optical components."""
    
    def __init__(self, first: OpticalComponent, second: OpticalComponent):
        self.first = first
        self.second = second
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply both components in sequence."""
        intermediate = self.first.apply(field)
        return self.second.apply(intermediate)
    
    def propagate(self, next_component: OpticalComponent) -> 'ChainedComponent':
        """Continue the chain."""
        return ChainedComponent(self, next_component)
```

### Component Implementations

```python
class Primary(OpticalComponent):
    """Primary mirror component."""
    
    def __init__(self, diameter: float, obscuration_ratio: float = 0.0):
        self.diameter = diameter
        self.obscuration_ratio = obscuration_ratio
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply pupil mask."""
        shape = field.shape
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = (shape[0] / 2, shape[1] / 2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r_norm = r / (min(shape) / 2)
        
        pupil = (r_norm <= 1.0) & (r_norm >= self.obscuration_ratio)
        return field * pupil.astype(field.dtype)

class Coronagraph(OpticalComponent):
    """Coronagraph component using yippy."""
    
    def __init__(self, yippy_coronagraph):
        self.yippy_model = yippy_coronagraph
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply coronagraph suppression using yippy."""
        # yippy handles all the physics
        return self.yippy_model.propagate(field)

class Filter(OpticalComponent):
    """Spectral filter component."""
    
    def __init__(self, center_wavelength: float, bandwidth: float, 
                 transmission: float = 0.9):
        self.center_wavelength = center_wavelength
        self.bandwidth = bandwidth
        self.transmission = transmission
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply spectral filtering."""
        # For monochromatic, just apply transmission
        # For polychromatic, would check wavelength
        return field * self.transmission

class Detector(OpticalComponent):
    """Detector component."""
    
    def __init__(self, shape: tuple[int, int], pixel_scale: float,
                 quantum_efficiency: float = 0.9, read_noise: float = 0.0,
                 dark_current: float = 0.0, exposure_time: float = 1.0):
        self.shape = shape
        self.pixel_scale = pixel_scale
        self.quantum_efficiency = quantum_efficiency
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.exposure_time = exposure_time
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Convert field to detected image."""
        # Convert to intensity
        intensity = np.abs(field)**2
        
        # Resample if needed
        if intensity.shape != self.shape:
            from scipy.ndimage import zoom
            zoom_factors = (self.shape[0] / intensity.shape[0],
                          self.shape[1] / intensity.shape[1])
            intensity = zoom(intensity, zoom_factors, order=1)
        
        # Apply detector effects
        photons = intensity * self.quantum_efficiency * self.exposure_time
        
        # Add noise (if not in JAX mode)
        if self.read_noise > 0 or self.dark_current > 0:
            # Handle noise addition based on backend
            signal = photons + self.dark_current * self.exposure_time
            # Add Poisson and read noise as appropriate
        else:
            signal = photons
        
        return signal
```

## Pipeline Wrapper

```python
class Pipeline:
    """Wrapper for optical component chains."""
    
    def __init__(self, component_chain: OpticalComponent):
        self.component_chain = component_chain
    
    def __call__(self, source: np.ndarray) -> np.ndarray:
        """Execute the pipeline on a source."""
        return self.component_chain.apply(source)
    
    def jit_compile(self):
        """Return a JAX-compiled version of this pipeline."""
        import jax
        
        # Convert to JAX-compatible function
        def jax_apply(source):
            return self.component_chain.apply(source)
        
        return jax.jit(jax_apply)
```

## Source-Specific Pipelines

### Telescope Configuration

```python
class TelescopeConfig:
    """Configuration for all telescope components."""
    
    def __init__(self, primary_diameter: float, coronagraph_path: str,
                 filter_wavelength: float, detector_shape: tuple[int, int],
                 pixel_scale: float):
        # Create components
        self.primary = Primary(primary_diameter, obscuration_ratio=0.14)
        self.coronagraph = Coronagraph(yippy.Coronagraph(coronagraph_path))
        self.filter = Filter(filter_wavelength, bandwidth=100e-9)
        self.detector = Detector(detector_shape, pixel_scale)
    
    def create_star_pipeline(self) -> Pipeline:
        """Create pipeline for star observation."""
        return Pipeline(
            self.primary.propagate(self.coronagraph)
                       .propagate(self.filter)
                       .propagate(self.detector)
        )
    
    def create_planet_pipeline(self) -> Pipeline:
        """Create pipeline for planet observation."""
        # Planets go through coronagraph
        return Pipeline(
            self.primary.propagate(self.coronagraph)
                       .propagate(self.filter)
                       .propagate(self.detector)
        )
    
    def create_reference_pipeline(self) -> Pipeline:
        """Create pipeline for reference star (no coronagraph)."""
        return Pipeline(
            self.primary.propagate(self.filter)
                       .propagate(self.detector)
        )
    
    def create_disk_pipeline(self) -> Pipeline:
        """Create pipeline for disk observation."""
        # Disk might have different handling
        return Pipeline(
            self.primary.propagate(self.coronagraph)
                       .propagate(self.filter)
                       .propagate(self.detector)
        )
```

### Multi-Source Observatory

```python
class Observatory:
    """Handles multiple source types efficiently."""
    
    def __init__(self, telescope_config: TelescopeConfig):
        self.config = telescope_config
        
        # Create pipelines for each source type
        self.star_pipeline = telescope_config.create_star_pipeline()
        self.planet_pipeline = telescope_config.create_planet_pipeline()
        self.reference_pipeline = telescope_config.create_reference_pipeline()
        self.disk_pipeline = telescope_config.create_disk_pipeline()
        
        # JAX-compiled versions
        self.star_pipeline_jax = self.star_pipeline.jit_compile()
        self.planet_pipeline_jax = self.planet_pipeline.jit_compile()
    
    def observe(self, sources: dict[str, np.ndarray], 
                use_jax: bool = False) -> dict[str, np.ndarray]:
        """
        Observe multiple sources.
        
        Args:
            sources: Dict mapping source type to array
                    e.g. {'star': star_array, 'planets': planet_arrays}
            use_jax: Whether to use JAX-compiled pipelines
        
        Returns:
            Dict mapping source type to output image
        """
        results = {}
        
        # Define expected shape for outputs
        output_shape = self.config.detector.shape
        
        # Process star
        if 'star' in sources and sources['star'] is not None:
            pipeline = self.star_pipeline_jax if use_jax else self.star_pipeline
            results['star'] = pipeline(sources['star'])
        else:
            results['star'] = np.zeros(output_shape)
        
        # Process planets (might be multiple)
        if 'planets' in sources and sources['planets'] is not None:
            if sources['planets'].ndim == 3:  # Multiple planets
                planet_images = []
                for i in range(sources['planets'].shape[0]):
                    pipeline = self.planet_pipeline_jax if use_jax else self.planet_pipeline
                    planet_images.append(pipeline(sources['planets'][i]))
                results['planets'] = np.stack(planet_images)
            else:  # Single planet
                pipeline = self.planet_pipeline_jax if use_jax else self.planet_pipeline
                results['planets'] = pipeline(sources['planets'])
        else:
            results['planets'] = np.zeros(output_shape)
        
        # Process reference star
        if 'reference' in sources and sources['reference'] is not None:
            pipeline = self.reference_pipeline_jax if use_jax else self.reference_pipeline
            results['reference'] = pipeline(sources['reference'])
        else:
            results['reference'] = np.zeros(output_shape)
        
        # Process disk
        if 'disk' in sources and sources['disk'] is not None:
            pipeline = self.disk_pipeline_jax if use_jax else self.disk_pipeline
            results['disk'] = pipeline(sources['disk'])
        else:
            results['disk'] = np.zeros(output_shape)
        
        return results
    
    def combine_sources(self, results: dict[str, np.ndarray]) -> np.ndarray:
        """Combine individual source images into final image."""
        combined = np.zeros_like(results['star'])
        
        for source_type, image in results.items():
            if image.ndim == 3:  # Multiple sources (e.g., planets)
                combined += np.sum(image, axis=0)
            else:
                combined += image
        
        return combined
```

## JAX Optimization

### Pure JAX Implementation

```python
import jax
import jax.numpy as jnp

class JAXPrimary:
    """JAX-optimized primary mirror."""
    
    def __init__(self, diameter: float, obscuration_ratio: float = 0.0):
        self.diameter = diameter
        self.obscuration_ratio = obscuration_ratio
        # Pre-compute pupil if shape is known
        self._pupil_cache = {}
    
    def get_pupil(self, shape: tuple[int, int]) -> jnp.ndarray:
        """Get or create pupil for given shape."""
        if shape not in self._pupil_cache:
            y, x = jnp.ogrid[:shape[0], :shape[1]]
            center = (shape[0] / 2, shape[1] / 2)
            r = jnp.sqrt((x - center[1])**2 + (y - center[0])**2)
            r_norm = r / (min(shape) / 2)
            pupil = (r_norm <= 1.0) & (r_norm >= self.obscuration_ratio)
            self._pupil_cache[shape] = pupil.astype(jnp.float32)
        return self._pupil_cache[shape]
    
    def apply(self, field: jnp.ndarray) -> jnp.ndarray:
        """Apply pupil to field."""
        pupil = self.get_pupil(field.shape)
        return field * pupil

# Create fully JAX-compatible pipeline
def create_jax_pipeline(config: TelescopeConfig):
    """Create a fully JAX-optimized pipeline."""
    
    # Convert components to JAX versions
    primary_jax = JAXPrimary(config.primary.diameter, 
                            config.primary.obscuration_ratio)
    
    @jax.jit
    def pipeline(source: jnp.ndarray) -> jnp.ndarray:
        # Apply primary
        field = primary_jax.apply(source)
        
        # Apply coronagraph (if yippy supports JAX)
        # field = coronagraph_jax.apply(field)
        
        # Apply filter
        field = field * config.filter.transmission
        
        # Convert to intensity and resize
        intensity = jnp.abs(field)**2
        
        # Simple detector model
        photons = intensity * config.detector.quantum_efficiency
        signal = photons * config.detector.exposure_time
        
        return signal
    
    return pipeline
```

### Batched Processing

```python
def create_batched_pipeline(config: TelescopeConfig):
    """Create pipeline that processes multiple sources efficiently."""
    
    base_pipeline = create_jax_pipeline(config)
    
    # Use vmap for batching
    batched_pipeline = jax.vmap(base_pipeline, in_axes=0, out_axes=0)
    
    return jax.jit(batched_pipeline)

# Usage
telescope = TelescopeConfig(6.5, 'coronagraphs/demo', 550e-9, (256, 256), 0.02)
batch_pipeline = create_batched_pipeline(telescope)

# Process multiple planets at once
planet_sources = jnp.stack([planet1, planet2, planet3])
planet_images = batch_pipeline(planet_sources)  # Shape: (3, 256, 256)
```

## Usage Examples

### Basic Usage

```python
# Configure telescope
telescope = TelescopeConfig(
    primary_diameter=6.5,
    coronagraph_path='input/coronagraphs/demo',
    filter_wavelength=550e-9,
    detector_shape=(256, 256),
    pixel_scale=0.02
)

# Create individual pipelines
star_pipeline = Pipeline(
    telescope.primary.propagate(telescope.coronagraph)
                    .propagate(telescope.filter)
                    .propagate(telescope.detector)
)

# Use the pipeline
star_image = star_pipeline(star_source)
```

### Multi-Source Observation

```python
# Create observatory
observatory = Observatory(telescope)

# Prepare sources
sources = {
    'star': load_star_source(),
    'planets': load_planet_sources(),  # Shape: (n_planets, size, size)
    'reference': load_reference_star(),
    'disk': None  # No disk in this observation
}

# Observe all sources
results = observatory.observe(sources, use_jax=True)

# Combine into final image
final_image = observatory.combine_sources(results)
```

### Custom Pipeline Creation

```python
# Create a custom pipeline for testing without coronagraph
test_pipeline = Pipeline(
    telescope.primary.propagate(telescope.filter)
                    .propagate(telescope.detector)
)

# Create a pipeline with custom detector settings
custom_detector = Detector(
    shape=(512, 512),
    pixel_scale=0.01,
    quantum_efficiency=0.95,
    read_noise=3.0,
    exposure_time=1000.0
)

custom_pipeline = Pipeline(
    telescope.primary.propagate(telescope.coronagraph)
                    .propagate(telescope.filter)
                    .propagate(custom_detector)
)
```

### Conditional Pipeline Building

```python
def build_pipeline(use_coronagraph: bool = True, 
                  use_filter: bool = True) -> Pipeline:
    """Build pipeline based on configuration."""
    
    # Start with primary
    chain = telescope.primary
    
    # Add coronagraph if requested
    if use_coronagraph:
        chain = chain.propagate(telescope.coronagraph)
    
    # Add filter if requested
    if use_filter:
        chain = chain.propagate(telescope.filter)
    
    # Always end with detector
    chain = chain.propagate(telescope.detector)
    
    return Pipeline(chain)
```

## Testing Strategy

### Unit Tests for Components

```python
def test_primary_component():
    """Test primary mirror component."""
    primary = Primary(diameter=6.5, obscuration_ratio=0.14)
    
    # Test with uniform field
    field = np.ones((256, 256))
    result = primary.apply(field)
    
    # Check pupil was applied
    assert result.shape == field.shape
    assert np.any(result == 0)  # Some obscuration
    assert np.any(result == 1)  # Some transmission

def test_propagate_chaining():
    """Test propagate method creates proper chain."""
    primary = Primary(6.5)
    filter = Filter(550e-9, 100e-9)
    detector = Detector((256, 256), 0.02)
    
    # Create chain
    chain = primary.propagate(filter).propagate(detector)
    
    # Test chain execution
    field = np.ones((256, 256))
    result = chain.apply(field)
    
    assert result.shape == (256, 256)
```

### Integration Tests

```python
def test_full_observatory():
    """Test complete multi-source observation."""
    telescope = TelescopeConfig(6.5, 'test/coronagraph', 550e-9, (128, 128), 0.02)
    observatory = Observatory(telescope)
    
    # Create test sources
    sources = {
        'star': create_test_star(),
        'planets': create_test_planets(n=3),
        'reference': create_test_reference(),
        'disk': None
    }
    
    # Run observation
    results = observatory.observe(sources)
    
    # Verify outputs
    assert results['star'].shape == (128, 128)
    assert results['planets'].shape == (3, 128, 128)
    assert results['reference'].shape == (128, 128)
    assert np.all(results['disk'] == 0)  # Should be zeros
    
    # Test combination
    combined = observatory.combine_sources(results)
    assert combined.shape == (128, 128)
    assert np.any(combined > 0)
```

### Performance Tests

```python
def benchmark_pipelines():
    """Compare performance of different implementations."""
    import time
    
    telescope = TelescopeConfig(6.5, 'coronagraphs/demo', 550e-9, (256, 256), 0.02)
    observatory = Observatory(telescope)
    
    # Test data
    sources = {
        'star': np.random.randn(256, 256),
        'planets': np.random.randn(10, 256, 256)
    }
    
    # Benchmark regular pipeline
    start = time.time()
    for _ in range(100):
        results = observatory.observe(sources, use_jax=False)
    regular_time = time.time() - start
    
    # Benchmark JAX pipeline
    start = time.time()
    for _ in range(100):
        results = observatory.observe(sources, use_jax=True)
    jax_time = time.time() - start
    
    print(f"Regular pipeline: {regular_time:.3f}s")
    print(f"JAX pipeline: {jax_time:.3f}s")
    print(f"Speedup: {regular_time/jax_time:.2f}x")
```

## Summary

This architecture provides:

1. **Clean API**: `Pipeline(primary.propagate(coronagraph).propagate(detector))`
2. **Source separation**: Different pipelines for different source types
3. **JAX optimization**: No conditionals in hot paths
4. **Consistent outputs**: Returns zeros for missing sources
5. **Flexibility**: Easy to create custom pipelines
6. **Performance**: Batched processing for multiple sources

The key insight is that by separating pipelines by source type, we avoid JAX-unfriendly conditionals while maintaining a clean, composable API.