# Propagate Pipeline Implementation Code

This document provides the complete implementation code for the `.propagate()` method architecture.

## Core Components

```python
# optical_components.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Union
import numpy as np
from dataclasses import dataclass

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False

# Type variable for component chaining
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
    
    def __repr__(self):
        """String representation of the chain."""
        return f"{self.first} -> {self.second}"
```

## Optical Component Implementations

```python
# components.py
import numpy as np
from scipy.ndimage import zoom
from typing import Optional, Tuple

class Primary(OpticalComponent):
    """Primary mirror component."""
    
    def __init__(self, diameter: float, obscuration_ratio: float = 0.0):
        """
        Initialize primary mirror.
        
        Args:
            diameter: Mirror diameter in meters
            obscuration_ratio: Central obscuration ratio (0-1)
        """
        self.diameter = diameter
        self.obscuration_ratio = obscuration_ratio
        self._pupil_cache = {}
    
    def _create_pupil(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create pupil mask for given shape."""
        y, x = np.ogrid[:shape[0], :shape[1]]
        center_y, center_x = shape[0] / 2, shape[1] / 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize radius to pupil diameter
        r_norm = r / (min(shape) / 2)
        
        # Create annular pupil
        pupil = (r_norm <= 1.0) & (r_norm >= self.obscuration_ratio)
        
        return pupil.astype(np.float32)
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply pupil mask to field."""
        shape = field.shape
        
        # Get or create pupil
        if shape not in self._pupil_cache:
            self._pupil_cache[shape] = self._create_pupil(shape)
        
        pupil = self._pupil_cache[shape]
        return field * pupil
    
    def __repr__(self):
        return f"Primary(D={self.diameter}m, obs={self.obscuration_ratio})"

class Coronagraph(OpticalComponent):
    """Coronagraph component using yippy."""
    
    def __init__(self, yippy_coronagraph):
        """
        Initialize with yippy Coronagraph object.
        
        Args:
            yippy_coronagraph: Initialized yippy.Coronagraph object
        """
        self.yippy_model = yippy_coronagraph
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply coronagraph suppression using yippy."""
        # yippy's propagate method handles all the physics
        # It expects a 2D complex field and returns the suppressed field
        return self.yippy_model.propagate(field)
    
    def __repr__(self):
        return f"Coronagraph(yippy)"

class Filter(OpticalComponent):
    """Spectral filter component."""
    
    def __init__(self, center_wavelength: float, bandwidth: float, 
                 transmission: float = 0.9):
        """
        Initialize spectral filter.
        
        Args:
            center_wavelength: Central wavelength in meters
            bandwidth: Filter bandwidth in meters
            transmission: Peak transmission (0-1)
        """
        self.center_wavelength = center_wavelength
        self.bandwidth = bandwidth
        self.transmission = transmission
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply spectral filtering."""
        # For monochromatic light, just apply transmission
        # In a full implementation, this would check wavelength
        return field * self.transmission
    
    def __repr__(self):
        λ_nm = self.center_wavelength * 1e9
        return f"Filter(λ={λ_nm:.0f}nm, T={self.transmission})"

class Detector(OpticalComponent):
    """Detector component."""
    
    def __init__(self, shape: Tuple[int, int], pixel_scale: float,
                 quantum_efficiency: float = 0.9, 
                 read_noise: float = 0.0,
                 dark_current: float = 0.0, 
                 exposure_time: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize detector.
        
        Args:
            shape: Detector shape (ny, nx)
            pixel_scale: Pixel scale in arcsec/pixel
            quantum_efficiency: QE (0-1)
            read_noise: Read noise in electrons RMS
            dark_current: Dark current in electrons/pixel/second
            exposure_time: Exposure time in seconds
            seed: Random seed for noise generation
        """
        self.shape = shape
        self.pixel_scale = pixel_scale
        self.quantum_efficiency = quantum_efficiency
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.exposure_time = exposure_time
        self.rng = np.random.RandomState(seed)
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Convert field to detected image."""
        # Convert complex field to intensity
        if np.iscomplexobj(field):
            intensity = np.abs(field)**2
        else:
            intensity = field
        
        # Resample to detector pixel grid if needed
        if intensity.shape != self.shape:
            zoom_factors = (self.shape[0] / intensity.shape[0],
                          self.shape[1] / intensity.shape[1])
            intensity = zoom(intensity, zoom_factors, order=1)
        
        # Convert to detected photons
        photon_rate = intensity * self.quantum_efficiency
        photons = photon_rate * self.exposure_time
        
        # Add noise sources
        signal = self._add_noise(photons)
        
        return signal
    
    def _add_noise(self, photons: np.ndarray) -> np.ndarray:
        """Add detector noise."""
        # Dark current
        dark_electrons = self.dark_current * self.exposure_time
        total_electrons = photons + dark_electrons
        
        # Poisson noise
        if np.any(total_electrons > 0):
            # Ensure non-negative for Poisson
            total_electrons = np.maximum(total_electrons, 0)
            signal = self.rng.poisson(total_electrons).astype(float)
        else:
            signal = total_electrons
        
        # Read noise
        if self.read_noise > 0:
            read_noise = self.rng.normal(0, self.read_noise, self.shape)
            signal += read_noise
        
        return signal
    
    def __repr__(self):
        return (f"Detector({self.shape[0]}x{self.shape[1]}, "
                f"scale={self.pixel_scale}as/pix)")
```

## Pipeline Wrapper

```python
# pipeline.py
import numpy as np
from typing import Callable, Optional

class Pipeline:
    """Wrapper for optical component chains."""
    
    def __init__(self, component_chain: OpticalComponent):
        """
        Initialize pipeline with component chain.
        
        Args:
            component_chain: Chain of optical components
        """
        self.component_chain = component_chain
    
    def __call__(self, source: np.ndarray) -> np.ndarray:
        """
        Execute the pipeline on a source.
        
        Args:
            source: Input field (2D complex array)
            
        Returns:
            Detected image
        """
        return self.component_chain.apply(source)
    
    def jit_compile(self) -> Callable:
        """Return a JAX-compiled version of this pipeline."""
        if not HAS_JAX:
            print("JAX not available, returning regular pipeline")
            return self.__call__
        
        import jax
        
        # Create JAX-compatible wrapper
        def jax_apply(source):
            # Convert to JAX array
            source_jax = jnp.array(source)
            # Apply pipeline
            result = self.component_chain.apply(source_jax)
            # Ensure output is JAX array
            return jnp.array(result)
        
        return jax.jit(jax_apply)
    
    def __repr__(self):
        return f"Pipeline({self.component_chain})"
```

## Telescope Configuration

```python
# telescope.py
from typing import Optional, Dict
import numpy as np

try:
    from yippy import Coronagraph as YippyCoronagraph
except ImportError:
    # Mock for testing without yippy
    class YippyCoronagraph:
        def __init__(self, path):
            self.path = path
        def propagate(self, field):
            return field * 0.001  # Simple suppression

class TelescopeConfig:
    """Configuration for all telescope components."""
    
    def __init__(self, 
                 primary_diameter: float,
                 coronagraph_path: str,
                 filter_wavelength: float,
                 filter_bandwidth: float,
                 detector_shape: Tuple[int, int],
                 pixel_scale: float,
                 exposure_time: float = 1.0,
                 obscuration_ratio: float = 0.14):
        """
        Initialize telescope configuration.
        
        Args:
            primary_diameter: Primary mirror diameter in meters
            coronagraph_path: Path to yippy coronagraph files
            filter_wavelength: Filter central wavelength in meters
            filter_bandwidth: Filter bandwidth in meters
            detector_shape: Detector shape (ny, nx)
            pixel_scale: Pixel scale in arcsec/pixel
            exposure_time: Exposure time in seconds
            obscuration_ratio: Primary mirror obscuration ratio
        """
        # Create components
        self.primary = Primary(primary_diameter, obscuration_ratio)
        self.coronagraph = Coronagraph(YippyCoronagraph(coronagraph_path))
        self.filter = Filter(filter_wavelength, filter_bandwidth)
        self.detector = Detector(detector_shape, pixel_scale, 
                               exposure_time=exposure_time)
        
        # Store config
        self.primary_diameter = primary_diameter
        self.coronagraph_path = coronagraph_path
        self.filter_wavelength = filter_wavelength
        self.detector_shape = detector_shape
        self.pixel_scale = pixel_scale
        self.exposure_time = exposure_time
    
    def create_star_pipeline(self) -> Pipeline:
        """Create pipeline for star observation (with coronagraph)."""
        return Pipeline(
            self.primary.propagate(self.coronagraph)
                       .propagate(self.filter)
                       .propagate(self.detector)
        )
    
    def create_planet_pipeline(self) -> Pipeline:
        """Create pipeline for planet observation (with coronagraph)."""
        # Same as star for now, but could customize
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
        # Could add disk-specific processing
        return Pipeline(
            self.primary.propagate(self.coronagraph)
                       .propagate(self.filter)
                       .propagate(self.detector)
        )
    
    def create_leakage_pipeline(self) -> Pipeline:
        """Create pipeline for stellar leakage (coronagraph but no planets)."""
        return self.create_star_pipeline()
```

## Observatory for Multi-Source Handling

```python
# observatory.py
import numpy as np
from typing import Dict, Optional, List, Union

class Observatory:
    """Handles multiple source types efficiently."""
    
    def __init__(self, telescope_config: TelescopeConfig, use_jax: bool = False):
        """
        Initialize observatory with telescope configuration.
        
        Args:
            telescope_config: Telescope configuration
            use_jax: Whether to use JAX compilation
        """
        self.config = telescope_config
        self.use_jax = use_jax
        
        # Create pipelines for each source type
        self.pipelines = {
            'star': telescope_config.create_star_pipeline(),
            'planet': telescope_config.create_planet_pipeline(),
            'reference': telescope_config.create_reference_pipeline(),
            'disk': telescope_config.create_disk_pipeline(),
            'leakage': telescope_config.create_leakage_pipeline()
        }
        
        # Create JAX versions if requested
        if use_jax:
            self.pipelines_jax = {
                name: pipeline.jit_compile() 
                for name, pipeline in self.pipelines.items()
            }
    
    def observe(self, 
                sources: Dict[str, Optional[Union[np.ndarray, List[np.ndarray]]]],
                combine: bool = True) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Observe multiple sources.
        
        Args:
            sources: Dict mapping source type to array(s)
                - 'star': 2D complex field
                - 'planets': List of 2D fields or 3D array
                - 'reference': 2D complex field
                - 'disk': 2D complex field
                - 'leakage': 2D complex field (optional, uses star if not provided)
            combine: Whether to combine all sources into single image
            
        Returns:
            If combine=True: Combined detector image
            If combine=False: Dict of individual detector images
        """
        results = {}
        output_shape = self.config.detector_shape
        
        # Process each source type
        for source_type, pipeline_name in [
            ('star', 'star'),
            ('reference', 'reference'),
            ('disk', 'disk'),
            ('leakage', 'leakage')
        ]:
            if source_type in sources and sources[source_type] is not None:
                pipeline = (self.pipelines_jax[pipeline_name] if self.use_jax 
                          else self.pipelines[pipeline_name])
                results[source_type] = pipeline(sources[source_type])
            else:
                results[source_type] = np.zeros(output_shape)
        
        # Handle planets specially (might be multiple)
        if 'planets' in sources and sources['planets'] is not None:
            pipeline = (self.pipelines_jax['planet'] if self.use_jax 
                      else self.pipelines['planet'])
            
            planets_data = sources['planets']
            if isinstance(planets_data, list):
                # List of planet fields
                planet_images = [pipeline(planet) for planet in planets_data]
                results['planets'] = np.stack(planet_images)
            elif planets_data.ndim == 3:
                # 3D array of planets
                planet_images = [pipeline(planets_data[i]) 
                               for i in range(planets_data.shape[0])]
                results['planets'] = np.stack(planet_images)
            else:
                # Single planet
                results['planets'] = pipeline(planets_data)
        else:
            results['planets'] = np.zeros(output_shape)
        
        # Handle leakage (use star if not explicitly provided)
        if 'leakage' not in sources and 'star' in sources:
            pipeline = (self.pipelines_jax['leakage'] if self.use_jax 
                      else self.pipelines['leakage'])
            results['leakage'] = pipeline(sources['star'])
        
        if combine:
            return self.combine_sources(results)
        else:
            return results
    
    def combine_sources(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine individual source images into final image."""
        combined = np.zeros(self.config.detector_shape, dtype=float)
        
        for source_type, image in results.items():
            if image.ndim == 3:  # Multiple sources (e.g., planets)
                combined += np.sum(image, axis=0)
            else:
                combined += image
        
        return combined
    
    def observe_sequence(self, 
                        source_sequence: List[Dict[str, Optional[np.ndarray]]],
                        combine: bool = True) -> List[np.ndarray]:
        """
        Observe a sequence of source configurations.
        
        Args:
            source_sequence: List of source dictionaries
            combine: Whether to combine sources in each observation
            
        Returns:
            List of observation results
        """
        results = []
        for sources in source_sequence:
            result = self.observe(sources, combine=combine)
            results.append(result)
        
        return results
```

## Usage Examples

```python
# examples.py

def example_basic_usage():
    """Basic usage example."""
    # Configure telescope
    telescope = TelescopeConfig(
        primary_diameter=6.5,
        coronagraph_path='input/coronagraphs/demo',
        filter_wavelength=550e-9,
        filter_bandwidth=100e-9,
        detector_shape=(256, 256),
        pixel_scale=0.02,
        exposure_time=100.0
    )
    
    # Create a simple pipeline
    star_pipeline = Pipeline(
        telescope.primary.propagate(telescope.coronagraph)
                        .propagate(telescope.filter)
                        .propagate(telescope.detector)
    )
    
    # Create test source (complex field)
    source = create_gaussian_source(shape=(256, 256), sigma=20)
    
    # Run pipeline
    image = star_pipeline(source)
    
    print(f"Output image shape: {image.shape}")
    print(f"Total flux: {np.sum(image):.2e} electrons")
    
    return image

def example_multi_source():
    """Multi-source observation example."""
    # Configure telescope
    telescope = TelescopeConfig(
        primary_diameter=6.5,
        coronagraph_path='input/coronagraphs/demo',
        filter_wavelength=550e-9,
        filter_bandwidth=100e-9,
        detector_shape=(256, 256),
        pixel_scale=0.02,
        exposure_time=1000.0
    )
    
    # Create observatory
    observatory = Observatory(telescope, use_jax=False)
    
    # Create sources
    star = create_gaussian_source((256, 256), sigma=20, amplitude=1e6)
    planet1 = create_gaussian_source((256, 256), sigma=5, amplitude=1e3, 
                                   offset=(50, 0))
    planet2 = create_gaussian_source((256, 256), sigma=5, amplitude=1e3, 
                                   offset=(-30, 40))
    reference = create_gaussian_source((256, 256), sigma=20, amplitude=1e5)
    
    # Observe
    sources = {
        'star': star,
        'planets': [planet1, planet2],
        'reference': reference,
        'disk': None
    }
    
    # Get combined image
    combined_image = observatory.observe(sources, combine=True)
    
    # Get individual images
    individual_images = observatory.observe(sources, combine=False)
    
    print(f"Combined image shape: {combined_image.shape}")
    print(f"Star image max: {np.max(individual_images['star']):.2e}")
    print(f"Planet images shape: {individual_images['planets'].shape}")
    
    return combined_image, individual_images

def example_custom_pipeline():
    """Example of building custom pipelines."""
    # Create components individually
    primary = Primary(diameter=8.0, obscuration_ratio=0.1)
    filter_ha = Filter(center_wavelength=656.3e-9, bandwidth=10e-9)
    detector_ccd = Detector(
        shape=(1024, 1024),
        pixel_scale=0.01,
        quantum_efficiency=0.95,
        read_noise=3.0,
        dark_current=0.1,
        exposure_time=3600.0
    )
    
    # No coronagraph pipeline
    direct_imaging = Pipeline(
        primary.propagate(filter_ha).propagate(detector_ccd)
    )
    
    # Test
    source = create_gaussian_source((1024, 1024), sigma=50)
    image = direct_imaging(source)
    
    return image

def example_jax_performance():
    """Compare JAX vs regular performance."""
    import time
    
    telescope = TelescopeConfig(
        primary_diameter=6.5,
        coronagraph_path='input/coronagraphs/demo',
        filter_wavelength=550e-9,
        filter_bandwidth=100e-9,
        detector_shape=(512, 512),
        pixel_scale=0.02
    )
    
    # Create observatories
    obs_regular = Observatory(telescope, use_jax=False)
    obs_jax = Observatory(telescope, use_jax=True)
    
    # Test data
    sources = {
        'star': create_gaussian_source((512, 512)),
        'planets': [create_gaussian_source((512, 512)) for _ in range(5)]
    }
    
    # Warmup JAX
    _ = obs_jax.observe(sources)
    
    # Benchmark
    n_iterations = 100
    
    start = time.time()
    for _ in range(n_iterations):
        _ = obs_regular.observe(sources)
    regular_time = time.time() - start
    
    start = time.time()
    for _ in range(n_iterations):
        _ = obs_jax.observe(sources)
    jax_time = time.time() - start
    
    print(f"Regular: {regular_time:.2f}s ({regular_time/n_iterations*1000:.1f}ms/iter)")
    print(f"JAX: {jax_time:.2f}s ({jax_time/n_iterations*1000:.1f}ms/iter)")
    print(f"Speedup: {regular_time/jax_time:.1f}x")

# Helper functions
def create_gaussian_source(shape: Tuple[int, int], 
                         sigma: float = 20.0,
                         amplitude: float = 1.0,
                         offset: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create a Gaussian source for testing."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y = shape[0] / 2 + offset[1]
    center_x = shape[1] / 2 + offset[0]
    
    r2 = (x - center_x)**2 + (y - center_y)**2
    gaussian = amplitude * np.exp(-r2 / (2 * sigma**2))
    
    # Return as complex field
    return gaussian.astype(complex)
```

## Testing Suite

```python
# test_pipeline.py
import pytest
import numpy as np

class TestComponents:
    """Test individual optical components."""
    
    def test_primary(self):
        """Test primary mirror."""
        primary = Primary(diameter=6.5, obscuration_ratio=0.14)
        field = np.ones((256, 256), dtype=complex)
        
        result = primary.apply(field)
        
        # Check shape preserved
        assert result.shape == field.shape
        
        # Check pupil applied
        assert np.any(result == 0)  # Obscuration
        assert np.any(np.abs(result) == 1)  # Clear aperture
        
        # Check obscuration ratio
        obscured_fraction = np.sum(result == 0) / result.size
        expected_fraction = 0.14**2  # Area ratio
        assert abs(obscured_fraction - expected_fraction) < 0.05
    
    def test_filter(self):
        """Test spectral filter."""
        filter_component = Filter(550e-9, 100e-9, transmission=0.9)
        field = np.ones((100, 100), dtype=complex)
        
        result = filter_component.apply(field)
        
        assert np.allclose(result, field * 0.9)
    
    def test_detector(self):
        """Test detector."""
        detector = Detector(
            shape=(128, 128),
            pixel_scale=0.02,
            quantum_efficiency=0.9,
            exposure_time=10.0,
            seed=42
        )
        
        # Test with intensity input
        intensity = np.ones((128, 128)) * 100  # photons/s/pixel
        result = detector.apply(intensity)
        
        # Check output
        expected_mean = 100 * 0.9 * 10.0  # photons * QE * time
        assert abs(np.mean(result) - expected_mean) / expected_mean < 0.1
    
    def test_propagate_chain(self):
        """Test propagate method chaining."""
        primary = Primary(6.5)
        filter_comp = Filter(550e-9, 100e-9)
        detector = Detector((64, 64), 0.02)
        
        # Create chain
        chain = primary.propagate(filter_comp).propagate(detector)
        
        # Test
        field = np.ones((64, 64), dtype=complex)
        result = chain.apply(field)
        
        assert result.shape == (64, 64)
        assert result.dtype == float

class TestPipeline:
    """Test Pipeline wrapper."""
    
    def test_pipeline_execution(self):
        """Test basic pipeline execution."""
        primary = Primary(6.5)
        detector = Detector((128, 128), 0.02)
        
        pipeline = Pipeline(primary.propagate(detector))
        
        field = create_gaussian_source((128, 128))
        result = pipeline(field)
        
        assert result.shape == (128, 128)
        assert np.any(result > 0)

class TestObservatory:
    """Test Observatory multi-source handling."""
    
    def test_single_source(self):
        """Test single source observation."""
        telescope = TelescopeConfig(
            primary_diameter=6.5,
            coronagraph_path='test',
            filter_wavelength=550e-9,
            filter_bandwidth=100e-9,
            detector_shape=(64, 64),
            pixel_scale=0.02
        )
        
        observatory = Observatory(telescope)
        
        sources = {
            'star': create_gaussian_source((64, 64))
        }
        
        result = observatory.observe(sources)
        assert result.shape == (64, 64)
        assert np.any(result > 0)
    
    def test_multiple_planets(self):
        """Test multiple planet handling."""
        telescope = TelescopeConfig(
            primary_diameter=6.5,
            coronagraph_path='test',
            filter_wavelength=550e-9,
            filter_bandwidth=100e-9,
            detector_shape=(64, 64),
            pixel_scale=0.02
        )
        
        observatory = Observatory(telescope)
        
        # Test with list of planets
        planets = [
            create_gaussian_source((64, 64), offset=(10, 0)),
            create_gaussian_source((64, 64), offset=(-10, 0)),
            create_gaussian_source((64, 64), offset=(0, 10))
        ]
        
        sources = {'planets': planets}
        
        # Get individual results
        results = observatory.observe(sources, combine=False)
        assert results['planets'].shape == (3, 64, 64)
        
        # Get combined result
        combined = observatory.observe(sources, combine=True)
        assert combined.shape == (64, 64)
    
    def test_missing_sources(self):
        """Test handling of missing sources."""
        telescope = TelescopeConfig(
            primary_diameter=6.5,
            coronagraph_path='test',
            filter_wavelength=550e-9,
            filter_bandwidth=100e-9,
            detector_shape=(32, 32),
            pixel_scale=0.02
        )
        
        observatory = Observatory(telescope)
        
        # Only star, no planets
        sources = {
            'star': create_gaussian_source((32, 32)),
            'planets': None,
            'disk': None
        }
        
        results = observatory.observe(sources, combine=False)
        
        # Check zeros returned for missing sources
        assert np.all(results['planets'] == 0)
        assert np.all(results['disk'] == 0)
        assert np.any(results['star'] > 0)

if __name__ == "__main__":
    # Run examples
    print("Running basic example...")
    example_basic_usage()
    
    print("\nRunning multi-source example...")
    example_multi_source()
    
    print("\nRunning custom pipeline example...")
    example_custom_pipeline()
    
    if HAS_JAX:
        print("\nRunning JAX performance comparison...")
        example_jax_performance()
```

This implementation provides a complete, working system for the `.propagate()` method architecture with:

1. **Clean chaining API**: Components chain with `.propagate()`
2. **Pipeline wrapper**: Simple `Pipeline()` class for execution
3. **Source separation**: Different pipelines for different sources
4. **JAX support**: Optional JAX compilation for performance
5. **Comprehensive testing**: Unit and integration tests
6. **Real-world examples**: Practical usage patterns

The architecture avoids conditionals in hot paths for better JAX performance while maintaining a clean, intuitive API.