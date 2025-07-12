"""
Integration tests for the coronagraphoto v2 architecture.

This module tests the end-to-end functionality of the new architecture,
verifying that all components work together correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from functools import partial
from astropy.time import Time
import astropy.units as u
import numpy as np

from coronagraphoto.observation import Target, Observation, ObservationSequence
from coronagraphoto.observatory import Observatory
from coronagraphoto.light_paths import (
    apply_primary, apply_coronagraph, apply_filter, apply_detector,
    PrimaryParams, CoronagraphParams, FilterParams, DetectorParams
)
from coronagraphoto.reduction import ReductionPipeline, ReferenceSubtract, Derotate, StackFrames


def test_end_to_end_adi_simulation():
    """
    Test a complete end-to-end ADI simulation.
    
    This test verifies that we can:
    1. Plan an ADI observation sequence
    2. Configure an observatory with light paths
    3. Execute the observation  
    4. Process the results through a reduction pipeline
    
    The test follows the complete physics pipeline from the original implementation.
    """
    
    # Step 1: Define hardware components following original physics
    primary_params = PrimaryParams(
        diameter=8 * u.m, 
        reflectivity=0.95,
        frac_obscured=0.2  # 20% central obscuration
    )
    coronagraph_params = CoronagraphParams(
        inner_working_angle=0.1 * u.arcsec,
        outer_working_angle=1.0 * u.arcsec,
        throughput=0.1,
        contrast=1e-10,
        pixel_scale=0.05,  # lambda/D per pixel
        npixels=128  # 128x128 pixel coronagraph
    )
    filter_params = FilterParams(
        central_wavelength=550 * u.nm,
        bandwidth=100 * u.nm,
        transmission=0.8,
        spectral_resolution=50  # R = λ/Δλ
    )
    detector_params = DetectorParams(
        pixel_scale=0.02 * u.arcsec / u.pix,
        read_noise=3 * u.electron,
        dark_current=0.01 * u.electron / u.s,
        quantum_efficiency=0.9,
        saturation_level=65535 * u.electron,
        cic_rate=0.001 * u.electron,
        shape=(512, 512)
    )
    
    # Step 2: Create light paths using partial functions
    science_path = [
        partial(apply_primary, params=primary_params),
        partial(apply_coronagraph, params=coronagraph_params),
        partial(apply_filter, params=filter_params),
        partial(apply_detector, params=detector_params),
    ]
    
    light_paths = {
        "science_imaging": science_path,
    }
    
    # Step 3: Create observatory
    observatory = Observatory(light_paths=light_paths)
    
    # Step 4: Plan observation sequence
    target = Target("test_scene.fits", name="Test System")
    start_time = Time("2024-01-01T00:00:00")
    
    obs_sequence = ObservationSequence.for_adi(
        target=target,
        path_name="science_imaging",
        n_exposures=4,
        exposure_time=300 * u.s,
        start_time=start_time,
        total_roll_angle=180 * u.deg
    )
    
    # Step 5: Execute observation
    try:
        raw_data = observatory.run(obs_sequence, seed=42)
        
        # Verify the data structure
        assert raw_data is not None
        assert 'observation' in raw_data.dims
        assert raw_data.sizes['observation'] == 4
        
        # Verify metadata
        assert raw_data.attrs['num_observations'] == 4
        assert 'coronagraphoto_version' in raw_data.attrs
        
        print("✓ Observatory execution successful")
        
    except Exception as e:
        print(f"✗ Observatory execution failed: {e}")
        raise
    
    # Step 6: Create reduction pipeline
    reduction_pipeline = ReductionPipeline([
        ReferenceSubtract(method="normalized_subtraction"),
        Derotate(),
        StackFrames(method="mean"),
    ])
    
    # Step 7: Process data
    try:
        final_data = reduction_pipeline.process(raw_data)
        
        # Verify processing
        assert final_data is not None
        assert 'processing_history' in final_data.attrs
        assert len(final_data.attrs['processing_history']) == 3
        
        print("✓ Reduction pipeline successful")
        
    except Exception as e:
        print(f"✗ Reduction pipeline failed: {e}")
        raise
    
    print("✓ End-to-end ADI simulation completed successfully!")
    return final_data


def test_rdi_simulation():
    """
    Test a complete RDI simulation with different light paths.
    """
    
    # Hardware components
    primary_params = PrimaryParams(diameter=8 * u.m)
    filter_params = FilterParams(
        central_wavelength=550 * u.nm,
        bandwidth=100 * u.nm
    )
    detector_params = DetectorParams(
        pixel_scale=0.02 * u.arcsec,
        read_noise=3 * u.electron,
        dark_current=0.01 * u.electron / u.s
    )
    coronagraph_params = CoronagraphParams(
        inner_working_angle=0.1 * u.arcsec,
        outer_working_angle=1.0 * u.arcsec
    )
    
    # Different light paths for science and reference
    science_path = [
        partial(apply_primary, params=primary_params),
        partial(apply_coronagraph, params=coronagraph_params),
        partial(apply_filter, params=filter_params),
        partial(apply_detector, params=detector_params),
    ]
    
    # Reference path without coronagraph
    reference_path = [
        partial(apply_primary, params=primary_params),
        partial(apply_filter, params=filter_params),
        partial(apply_detector, params=detector_params),
    ]
    
    light_paths = {
        "science_imaging": science_path,
        "reference_imaging": reference_path,
    }
    
    observatory = Observatory(light_paths=light_paths)
    
    # Create targets
    science_target = Target("science_scene.fits", name="Science Target")
    ref_target = Target("ref_scene.fits", name="Reference Target")
    
    # Plan RDI sequence
    obs_sequence = ObservationSequence.for_rdi(
        science_target=science_target,
        ref_target=ref_target,
        science_path_name="science_imaging",
        ref_path_name="reference_imaging",
        exposure_time=300 * u.s,
        start_time=Time("2024-01-01T00:00:00")
    )
    
    # Execute
    try:
        raw_data = observatory.run(obs_sequence, seed=42)
        
        # Verify we have both science and reference data
        assert raw_data.sizes['observation'] == 2
        
        print("✓ RDI simulation successful")
        
    except Exception as e:
        print(f"✗ RDI simulation failed: {e}")
        raise
    
    return raw_data


def test_parameter_validation():
    """
    Test that parameter validation works correctly.
    """
    
    # Test invalid primary parameters
    try:
        PrimaryParams(diameter=-1 * u.m)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Primary parameter validation working")
    
    # Test invalid coronagraph parameters
    try:
        CoronagraphParams(
            inner_working_angle=1.0 * u.arcsec,
            outer_working_angle=0.5 * u.arcsec  # Smaller than inner
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Coronagraph parameter validation working")
    
    # Test invalid detector parameters
    try:
        DetectorParams(
            pixel_scale=0.02 * u.arcsec,
            read_noise=-1 * u.electron,  # Negative noise
            dark_current=0.01 * u.electron / u.s
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Detector parameter validation working")


def test_light_path_functions():
    """
    Test that individual light path functions work correctly.
    """
    from coronagraphoto.data_models import IntermediateData, PropagationContext
    
    # Create test data
    wavelengths = np.linspace(500, 600, 10) * u.nm
    flux = np.ones(10) * 1e6
    
    data = IntermediateData.from_star_spectrum(wavelengths, flux)
    
    # Create test context
    context = PropagationContext(
        time=Time("2024-01-01T00:00:00"),
        wavelength=550 * u.nm,
        bandpass_slice=10 * u.nm,
        time_step=300 * u.s,
        rng_key=42
    )
    
    # Test primary function
    primary_params = PrimaryParams(diameter=8 * u.m)
    result = apply_primary(data, primary_params, context)
    
    # Flux should be increased by collecting area
    collecting_area = np.pi * (4 * u.m)**2
    expected_flux = flux * 0.95 * collecting_area.to(u.m**2).value
    
    assert np.allclose(result.star_flux.values, expected_flux)
    print("✓ Primary function working")
    
    # Test coronagraph function
    coronagraph_params = CoronagraphParams(
        inner_working_angle=0.1 * u.arcsec,
        outer_working_angle=1.0 * u.arcsec
    )
    result = apply_coronagraph(result, coronagraph_params, context)
    
    # Star flux should be suppressed
    assert np.all(result.star_flux.values < expected_flux)
    print("✓ Coronagraph function working")


if __name__ == "__main__":
    print("Running coronagraphoto v2 integration tests...")
    print("=" * 50)
    
    try:
        test_parameter_validation()
        test_light_path_functions()
        test_end_to_end_adi_simulation()
        test_rdi_simulation()
        
        print("=" * 50)
        print("✓ All integration tests passed!")
        
    except Exception as e:
        print("=" * 50)
        print(f"✗ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)