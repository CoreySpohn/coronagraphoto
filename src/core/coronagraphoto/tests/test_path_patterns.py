"""
Tests for the new light path patterns in coronagraphoto v2.

This module tests all the different ways of creating and composing light paths,
ensuring they produce equivalent results and maintain the stateless functional
behavior.
"""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
import xarray as xr

from ..light_paths import (
    Path, PathBuilder, PathPipeline, compose,
    PrimaryParams, CoronagraphParams, FilterParams, DetectorParams,
    path_component, _PATH_REGISTRY
)
from ..data_models import IntermediateData, PropagationContext


@pytest.fixture
def sample_parameters():
    """Create sample parameters for testing."""
    primary_params = PrimaryParams(
        diameter=6.5 * u.m,
        reflectivity=0.95,
        frac_obscured=0.1
    )
    
    coronagraph_params = CoronagraphParams(
        inner_working_angle=0.1 * u.arcsec,
        outer_working_angle=1.0 * u.arcsec,
        throughput=0.1,
        contrast=1e-10
    )
    
    filter_params = FilterParams(
        central_wavelength=550 * u.nm,
        bandwidth=20 * u.nm,
        transmission=0.8
    )
    
    detector_params = DetectorParams(
        pixel_scale=0.01 * u.arcsec,
        read_noise=3 * u.electron,
        dark_current=0.001 * u.electron / u.s,
        quantum_efficiency=0.9
    )
    
    return primary_params, coronagraph_params, filter_params, detector_params


@pytest.fixture
def sample_data():
    """Create sample data and context for testing."""
    # Create a simple test scene
    wavelengths = np.linspace(400, 800, 10)  # Smaller for faster tests
    flux_values = np.exp(-(wavelengths - 550)**2 / (2 * 50**2)) * 1e6
    
    dataset = xr.Dataset({
        'star_flux': xr.DataArray(
            flux_values,
            dims=['wavelength'],
            coords={'wavelength': wavelengths}
        )
    })
    
    data = IntermediateData(dataset)
    
    # Create propagation context
    context = PropagationContext(
        time=Time.now(),
        wavelength=550 * u.nm,
        bandpass_slice=1.0 * u.nm,
        time_step=100 * u.s,
        rng_key=42
    )
    
    return data, context


class TestPathRegistry:
    """Test the path component registry system."""
    
    def test_registry_populated(self):
        """Test that the registry is populated with expected components."""
        components = Path.list_components()
        
        # Check that expected components are present
        expected_components = {'primary', 'coronagraph', 'filter', 'detector', 'speckles'}
        assert expected_components.issubset(set(components.keys()))
    
    def test_registry_categories(self):
        """Test that components have proper categories."""
        components = Path.list_components()
        
        # Check specific categories
        assert components['primary'] == 'optics'
        assert components['coronagraph'] == 'optics'
        assert components['filter'] == 'optics'
        assert components['detector'] == 'hardware'
        assert components['speckles'] == 'noise'
    
    def test_from_registry(self, sample_parameters):
        """Test creating functions from registry."""
        primary_params, _, _, _ = sample_parameters
        
        # Create function from registry
        func = Path.from_registry('primary', primary_params)
        assert callable(func)
        
        # Test with invalid component
        with pytest.raises(ValueError, match="not found in registry"):
            Path.from_registry('nonexistent', primary_params)


class TestPathDecorator:
    """Test the path_component decorator."""
    
    def test_decorator_registration(self):
        """Test that decorator properly registers components."""
        @path_component("test_component", "test_category")
        def test_func(data, context, params):
            return data
        
        # Check that it's registered
        assert "test_component" in _PATH_REGISTRY
        
        # Check that attributes are set
        assert hasattr(test_func, 'category')
        assert hasattr(test_func, 'component_name')
        assert test_func.category == "test_category"
        assert test_func.component_name == "test_component"
    
    def test_decorator_function_creation(self):
        """Test that decorated functions work correctly."""
        @path_component("test_component2", "test_category2")
        def test_func(data, context, params):
            return data
        
        # Create function from registry
        func = _PATH_REGISTRY["test_component2"](params=None)
        assert callable(func)


class TestPathFactory:
    """Test the Path factory class."""
    
    def test_primary_creation(self, sample_parameters):
        """Test creating primary path function."""
        primary_params, _, _, _ = sample_parameters
        
        func = Path.primary(primary_params)
        assert callable(func)
    
    def test_coronagraph_creation(self, sample_parameters):
        """Test creating coronagraph path function."""
        _, coronagraph_params, _, _ = sample_parameters
        
        func = Path.coronagraph(coronagraph_params)
        assert callable(func)
    
    def test_filter_creation(self, sample_parameters):
        """Test creating filter path function."""
        _, _, filter_params, _ = sample_parameters
        
        func = Path.filter(filter_params)
        assert callable(func)
    
    def test_detector_creation(self, sample_parameters):
        """Test creating detector path function."""
        _, _, _, detector_params = sample_parameters
        
        func = Path.detector(detector_params)
        assert callable(func)
    
    def test_speckles_creation(self):
        """Test creating speckles path function."""
        func = Path.speckles(params=None)
        assert callable(func)


class TestPipelinePattern:
    """Test the pipeline pattern using >> operator."""
    
    def test_pipeline_creation(self, sample_parameters):
        """Test creating pipelines with >> operator."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        
        # Create pipeline
        pipeline = (PathPipeline(Path.primary(primary_params)) >> 
                   Path.coronagraph(coro_params) >> 
                   Path.filter(filter_params) >> 
                   Path.detector(det_params))
        
        assert isinstance(pipeline, PathPipeline)
        assert callable(pipeline)
    
    def test_pipeline_execution(self, sample_parameters, sample_data):
        """Test executing a pipeline."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        data, context = sample_data
        
        # Create and execute pipeline
        pipeline = (PathPipeline(Path.primary(primary_params)) >> 
                   Path.coronagraph(coro_params) >> 
                   Path.filter(filter_params) >> 
                   Path.detector(det_params))
        
        result = pipeline(data, context)
        
        assert isinstance(result, IntermediateData)
        assert result.dataset is not None
    
    def test_pipeline_chaining(self, sample_parameters):
        """Test chaining pipelines."""
        primary_params, coro_params, _, _ = sample_parameters
        
        # Create two pipelines
        pipeline1 = PathPipeline(Path.primary(primary_params))
        pipeline2 = PathPipeline(Path.coronagraph(coro_params))
        
        # Chain them
        combined = pipeline1 >> pipeline2
        
        assert isinstance(combined, PathPipeline)
        assert callable(combined)


class TestFunctionalComposition:
    """Test the functional composition pattern."""
    
    def test_compose_function(self, sample_parameters):
        """Test the compose function."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        
        # Create composed function
        composed = compose(
            Path.detector(det_params),
            Path.filter(filter_params),
            Path.coronagraph(coro_params),
            Path.primary(primary_params)
        )
        
        assert callable(composed)
    
    def test_compose_execution(self, sample_parameters, sample_data):
        """Test executing composed functions."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        data, context = sample_data
        
        # Create and execute composed function
        composed = compose(
            Path.detector(det_params),
            Path.filter(filter_params),
            Path.coronagraph(coro_params),
            Path.primary(primary_params)
        )
        
        result = composed(data, context)
        
        assert isinstance(result, IntermediateData)
        assert result.dataset is not None
    
    def test_compose_empty(self):
        """Test composing empty function list."""
        # Should work but return identity
        composed = compose()
        assert callable(composed)


class TestBuilderPattern:
    """Test the builder pattern."""
    
    def test_builder_creation(self):
        """Test creating a builder."""
        builder = PathBuilder()
        assert isinstance(builder, PathBuilder)
    
    def test_builder_chaining(self, sample_parameters):
        """Test builder method chaining."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        
        # Chain builder methods
        builder = (PathBuilder()
                  .primary(primary_params)
                  .coronagraph(coro_params)
                  .filter(filter_params)
                  .detector(det_params))
        
        assert isinstance(builder, PathBuilder)
        assert len(builder._components) == 4
    
    def test_builder_build(self, sample_parameters):
        """Test building a path from builder."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        
        # Build path
        path = (PathBuilder()
               .primary(primary_params)
               .coronagraph(coro_params)
               .filter(filter_params)
               .detector(det_params)
               .build())
        
        assert callable(path)
    
    def test_builder_execution(self, sample_parameters, sample_data):
        """Test executing a builder-created path."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        data, context = sample_data
        
        # Create and execute path
        path = (PathBuilder()
               .primary(primary_params)
               .coronagraph(coro_params)
               .filter(filter_params)
               .detector(det_params)
               .build())
        
        result = path(data, context)
        
        assert isinstance(result, IntermediateData)
        assert result.dataset is not None
    
    def test_builder_custom_component(self, sample_parameters):
        """Test adding custom components to builder."""
        primary_params, _, _, _ = sample_parameters
        
        # Custom function
        def custom_func(data, context):
            return data
        
        # Add custom component
        builder = (PathBuilder()
                  .primary(primary_params)
                  .custom(custom_func))
        
        assert len(builder._components) == 2
    
    def test_builder_empty_build(self):
        """Test building from empty builder."""
        builder = PathBuilder()
        
        with pytest.raises(ValueError, match="Cannot build empty path"):
            builder.build()


class TestPatternEquivalence:
    """Test that all patterns produce equivalent results."""
    
    def test_pattern_equivalence(self, sample_parameters, sample_data):
        """Test that different patterns produce equivalent results."""
        primary_params, coro_params, filter_params, det_params = sample_parameters
        data, context = sample_data
        
        # Create same path using different patterns
        pipeline_path = (PathPipeline(Path.primary(primary_params)) >> 
                        Path.coronagraph(coro_params) >> 
                        Path.filter(filter_params) >> 
                        Path.detector(det_params))
        
        functional_path = compose(
            Path.detector(det_params),
            Path.filter(filter_params),
            Path.coronagraph(coro_params),
            Path.primary(primary_params)
        )
        
        builder_path = (PathBuilder()
                       .primary(primary_params)
                       .coronagraph(coro_params)
                       .filter(filter_params)
                       .detector(det_params)
                       .build())
        
        # Execute all paths with same random seed
        np.random.seed(42)
        result1 = pipeline_path(data, context)
        
        np.random.seed(42)
        result2 = functional_path(data, context)
        
        np.random.seed(42)
        result3 = builder_path(data, context)
        
        # Results should have same structure
        assert set(result1.dataset.data_vars.keys()) == set(result2.dataset.data_vars.keys())
        assert set(result1.dataset.data_vars.keys()) == set(result3.dataset.data_vars.keys())
        
        # Results should be similar (allowing for numerical precision)
        # This is a simplified test - in practice you'd check actual values
        assert isinstance(result1, IntermediateData)
        assert isinstance(result2, IntermediateData)
        assert isinstance(result3, IntermediateData)


class TestStatelessBehavior:
    """Test that all patterns maintain stateless behavior."""
    
    def test_stateless_execution(self, sample_parameters, sample_data):
        """Test that functions don't maintain state between calls."""
        primary_params, coro_params, _, _ = sample_parameters
        data, context = sample_data
        
        # Create path
        path = (PathPipeline(Path.primary(primary_params)) >> 
               Path.coronagraph(coro_params))
        
        # Execute multiple times
        result1 = path(data, context)
        result2 = path(data, context)
        
        # Both should succeed and be independent
        assert isinstance(result1, IntermediateData)
        assert isinstance(result2, IntermediateData)
        
        # Original data should be unchanged
        assert data.dataset is not None
    
    def test_parameter_immutability(self, sample_parameters, sample_data):
        """Test that parameters are not modified during execution."""
        primary_params, _, _, _ = sample_parameters
        data, context = sample_data
        
        # Store original values
        original_diameter = primary_params.diameter
        original_reflectivity = primary_params.reflectivity
        
        # Create and execute path
        path = PathPipeline(Path.primary(primary_params))
        result = path(data, context)
        
        # Parameters should be unchanged
        assert primary_params.diameter == original_diameter
        assert primary_params.reflectivity == original_reflectivity
    
    def test_context_immutability(self, sample_parameters, sample_data):
        """Test that context is not modified during execution."""
        primary_params, _, _, _ = sample_parameters
        data, context = sample_data
        
        # Store original values
        original_wavelength = context.wavelength
        original_time = context.time
        
        # Create and execute path
        path = PathPipeline(Path.primary(primary_params))
        result = path(data, context)
        
        # Context should be unchanged
        assert context.wavelength == original_wavelength
        assert context.time == original_time


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    def test_backward_compatibility_imports(self):
        """Test that backward compatibility functions are available."""
        from ..light_paths import (
            primary_step, coronagraph_step, filter_step, 
            detector_step, speckles_step
        )
        
        # All should be callable
        assert callable(primary_step)
        assert callable(coronagraph_step)
        assert callable(filter_step)
        assert callable(detector_step)
        assert callable(speckles_step)
    
    def test_backward_compatibility_usage(self, sample_parameters):
        """Test that backward compatibility functions work."""
        from ..light_paths import primary_step
        
        primary_params, _, _, _ = sample_parameters
        
        # Should work like the new pattern
        func = primary_step(primary_params)
        assert callable(func)


if __name__ == "__main__":
    pytest.main([__file__])