"""
Example usage of the coronagraphoto v2 architecture.

This script demonstrates how to use the new functional architecture
to plan and execute coronagraph observations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

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


def main():
    """
    Demonstrate the coronagraphoto v2 architecture with an end-to-end example.
    """
    
    print("Coronagraphoto v2 Architecture Demo")
    print("=" * 40)
    
    print("\n1. Defining Hardware Components")
    print("-" * 30)
    
    # Define the hardware components
    primary_params = PrimaryParams(
        diameter=8 * u.m,
        reflectivity=0.95,
        temperature=280 * u.K
    )
    print(f"Primary: {primary_params.diameter} diameter, {primary_params.reflectivity} reflectivity")
    
    coronagraph_params = CoronagraphParams(
        inner_working_angle=0.1 * u.arcsec,
        outer_working_angle=1.0 * u.arcsec,
        throughput=0.1,
        contrast=1e-10
    )
    print(f"Coronagraph: {coronagraph_params.inner_working_angle} IWA, {coronagraph_params.contrast} contrast")
    
    filter_params = FilterParams(
        central_wavelength=550 * u.nm,
        bandwidth=100 * u.nm,
        transmission=0.8
    )
    print(f"Filter: {filter_params.central_wavelength} ± {filter_params.bandwidth/2}")
    
    detector_params = DetectorParams(
        pixel_scale=0.02 * u.arcsec,
        read_noise=3 * u.electron,
        dark_current=0.01 * u.electron / u.s,
        quantum_efficiency=0.9
    )
    print(f"Detector: {detector_params.pixel_scale} pixel scale, {detector_params.read_noise} RN")
    
    print("\n2. Creating Light Paths")
    print("-" * 30)
    
    # Create light paths as sequences of pure functions
    # Use wrapper functions for proper parameter binding
    def primary_step(data, context):
        return apply_primary(data, primary_params, context)
    
    def coronagraph_step(data, context):
        return apply_coronagraph(data, coronagraph_params, context)
        
    def filter_step(data, context):
        return apply_filter(data, filter_params, context)
        
    def detector_step(data, context):
        return apply_detector(data, detector_params, context)
    
    science_path = [
        primary_step,
        coronagraph_step,
        filter_step,
        detector_step,
    ]
    
    # Alternative path without coronagraph for reference observations
    reference_path = [
        primary_step,
        filter_step,
        detector_step,
    ]
    
    light_paths = {
        "science_imaging": science_path,
        "reference_imaging": reference_path,
    }
    
    print(f"Science path: {len(science_path)} steps")
    print(f"Reference path: {len(reference_path)} steps")
    
    print("\n3. Configuring Observatory")
    print("-" * 30)
    
    # Create the observatory with available light paths
    observatory = Observatory(light_paths=light_paths)
    print(f"Observatory configured with {len(light_paths)} light paths")
    
    print("\n4. Planning Observations")
    print("-" * 30)
    
    # Create targets
    science_target = Target("input/scenes/more_pix.fits", name="ExoVista System")
    ref_target = Target("input/scenes/ref_star.fits", name="Reference Star")
    
    print(f"Science target: {science_target.name}")
    print(f"Reference target: {ref_target.name}")
    
    # Plan an ADI observation sequence
    start_time = Time("2024-01-01T00:00:00")
    
    adi_sequence = ObservationSequence.for_adi(
        target=science_target,
        path_name="science_imaging",
        n_exposures=8,
        exposure_time=300 * u.s,
        start_time=start_time,
        total_roll_angle=360 * u.deg
    )
    
    print(f"ADI sequence: {len(adi_sequence)} exposures")
    print(f"Total exposure time: {adi_sequence.total_exposure_time}")
    print(f"Total duration: {adi_sequence.total_duration}")
    
    # Also plan an RDI sequence for comparison
    rdi_sequence = ObservationSequence.for_rdi(
        science_target=science_target,
        ref_target=ref_target,
        science_path_name="science_imaging",
        ref_path_name="reference_imaging",
        exposure_time=300 * u.s,
        start_time=start_time + 1 * u.hour
    )
    
    print(f"RDI sequence: {len(rdi_sequence)} exposures")
    
    print("\n5. Executing Observations")
    print("-" * 30)
    
    # Execute the ADI sequence
    print("Executing ADI sequence...")
    try:
        adi_data = observatory.run(adi_sequence, seed=42)
        print(f"✓ ADI execution successful")
        print(f"  Data shape: {dict(adi_data.sizes)}")
        print(f"  Metadata keys: {list(adi_data.attrs.keys())}")
        
    except Exception as e:
        print(f"✗ ADI execution failed: {e}")
        return
    
    # Execute the RDI sequence
    print("Executing RDI sequence...")
    try:
        rdi_data = observatory.run(rdi_sequence, seed=42)
        print(f"✓ RDI execution successful")
        print(f"  Data shape: {dict(rdi_data.sizes)}")
        
    except Exception as e:
        print(f"✗ RDI execution failed: {e}")
        return
    
    print("\n6. Post-Processing")
    print("-" * 30)
    
    # Create reduction pipelines
    adi_pipeline = ReductionPipeline([
        Derotate(reference_angle=0 * u.deg),
        StackFrames(method="mean"),
        # BackgroundSubtract(method="annulus"),
        # ContrastCurve(separations=np.linspace(0.1, 1.0, 10), sigma_level=5.0),
    ])
    
    rdi_pipeline = ReductionPipeline([
        ReferenceSubtract(method="normalized_subtraction"),
        StackFrames(method="mean"),
    ])
    
    print(f"ADI pipeline: {len(adi_pipeline)} steps")
    print(f"RDI pipeline: {len(rdi_pipeline)} steps")
    
    # Process the data
    print("Processing ADI data...")
    try:
        adi_final = adi_pipeline.process(adi_data)
        print(f"✓ ADI processing successful")
        print(f"  Processing history: {adi_final.attrs.get('processing_history', [])}")
        
    except Exception as e:
        print(f"✗ ADI processing failed: {e}")
        return
    
    print("Processing RDI data...")
    try:
        rdi_final = rdi_pipeline.process(rdi_data)
        print(f"✓ RDI processing successful")
        print(f"  Processing history: {rdi_final.attrs.get('processing_history', [])}")
        
    except Exception as e:
        print(f"✗ RDI processing failed: {e}")
        return
    
    print("\n7. Results Summary")
    print("-" * 30)
    
    print(f"ADI final data:")
    print(f"  Variables: {list(adi_final.data_vars.keys())}")
    print(f"  Dimensions: {dict(adi_final.sizes)}")
    
    print(f"RDI final data:")
    print(f"  Variables: {list(rdi_final.data_vars.keys())}")
    print(f"  Dimensions: {dict(rdi_final.sizes)}")
    
    print("\n" + "=" * 40)
    print("✓ Coronagraphoto v2 demo completed successfully!")
    print("\nKey advantages of the new architecture:")
    print("- Pure functional light paths for easy testing and optimization")
    print("- Flexible observation planning with builder patterns")
    print("- Modular post-processing pipeline")
    print("- Clear separation of concerns")
    print("- Ready for JAX acceleration in future phases")
    
    return adi_final, rdi_final


if __name__ == "__main__":
    main()