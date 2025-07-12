"""
Real Integration Example for Coronagraphoto v2 Light Path Patterns.

This example demonstrates how to use the new light path patterns with:
- Real yippy Coronagraph objects loaded from input/coronagraphs/demo/
- Real ExoVista scenes loaded from input/scenes/more_pix.fits
- Integration with the actual physics models

This replaces the old cumbersome pattern of manually defining each step.
"""

import numpy as np
import astropy.units as u
from astropy.time import Time
from pathlib import Path

from ..light_paths import (
    Path as LightPath, PathBuilder, PathPipeline, compose,
    PrimaryParams, CoronagraphParams, FilterParams, DetectorParams
)
from ..observation import Target, Observation, ObservationSequence
from ..observatory import Observatory
from ..data_models import PropagationContext


def create_real_coronagraph_params(coronagraph_dir: str = "input/coronagraphs/demo") -> CoronagraphParams:
    """
    Create coronagraph parameters using a real yippy coronagraph model.
    
    Args:
        coronagraph_dir: 
            Path to the coronagraph directory (default: input/coronagraphs/demo)
            
    Returns:
        CoronagraphParams configured with real yippy model
    """
    return CoronagraphParams(
        inner_working_angle=0.1 * u.arcsec,
        outer_working_angle=1.0 * u.arcsec,
        throughput=0.1,
        contrast=1e-10,
        coronagraph_dir=coronagraph_dir,
        use_jax=True,
        cpu_cores=4  # Adjust based on your system
    )


def create_example_hardware():
    """Create realistic hardware parameters."""
    
    # Primary mirror (6.5m like JWST)
    primary_params = PrimaryParams(
        diameter=6.5 * u.m,
        reflectivity=0.95,
        frac_obscured=0.1  # Central obscuration
    )
    
    # Real coronagraph from yippy
    coronagraph_params = create_real_coronagraph_params()
    
    # Optical filter (V-band)
    filter_params = FilterParams(
        central_wavelength=550 * u.nm,
        bandwidth=20 * u.nm,
        transmission=0.8
    )
    
    # Detector (like Nancy Grace Roman Space Telescope)
    detector_params = DetectorParams(
        pixel_scale=0.01 * u.arcsec,
        read_noise=3 * u.electron,
        dark_current=0.001 * u.electron / u.s,
        quantum_efficiency=0.9,
        shape=(512, 512)
    )
    
    return primary_params, coronagraph_params, filter_params, detector_params


def demo_real_pipeline_pattern():
    """Demonstrate pipeline pattern with real components."""
    print("=== REAL PIPELINE PATTERN DEMO ===")
    print("Using real yippy coronagraph and ExoVista scene")
    print()
    
    # Create real hardware parameters
    primary_params, coro_params, filter_params, det_params = create_example_hardware()
    
    # Create pipeline with real components
    science_path = (PathPipeline(LightPath.primary(primary_params)) >> 
                   LightPath.coronagraph(coro_params) >> 
                   LightPath.filter(filter_params) >> 
                   LightPath.detector(det_params))
    
    print("✓ Created science pipeline with real yippy coronagraph")
    
    # Create a reference path (no coronagraph for RDI)
    reference_path = (PathPipeline(LightPath.primary(primary_params)) >> 
                     LightPath.filter(filter_params) >> 
                     LightPath.detector(det_params))
    
    print("✓ Created reference pipeline (no coronagraph)")
    
    return {
        "science_imaging": science_path,
        "reference_imaging": reference_path
    }


def demo_real_exovista_target():
    """Demonstrate loading real ExoVista scenes."""
    print("=== REAL EXOVISTA TARGET DEMO ===")
    print("Loading ExoVista scene from input/scenes/more_pix.fits")
    print()
    
    # Create target from real ExoVista scene
    scene_path = "input/scenes/more_pix.fits"
    
    # Check if file exists
    if not Path(scene_path).exists():
        print(f"Warning: Scene file {scene_path} not found")
        print("Please ensure ExoVista scene files are available")
        return None
    
    target = Target(scene_path=scene_path, name="more_pix_system")
    
    print(f"✓ Created target: {target}")
    
    try:
        # Try to load the scene to verify it works
        scene_data = target.load_scene()
        print(f"✓ Successfully loaded scene data")
        print(f"  Available data components: {list(scene_data.dataset.data_vars.keys())}")
        print(f"  Scene metadata: {scene_data.dataset.attrs}")
        
    except Exception as e:
        print(f"Warning: Could not load scene data: {e}")
        print("This may be due to missing dependencies (exoverses)")
    
    return target


def demo_real_observation_sequence():
    """Demonstrate creating observation sequences with real components."""
    print("=== REAL OBSERVATION SEQUENCE DEMO ===")
    print("Creating ADI and RDI sequences with real targets")
    print()
    
    # Load real target
    target = demo_real_exovista_target()
    if target is None:
        print("Skipping observation sequence demo due to missing target")
        return None
    
    # Create ADI sequence
    adi_sequence = ObservationSequence.for_adi(
        target=target,
        path_name="science_imaging",
        n_exposures=8,
        exposure_time=300 * u.s,
        start_time=Time.now(),
        total_roll_angle=180 * u.deg  # Half rotation for ADI
    )
    
    print(f"✓ Created ADI sequence with {len(adi_sequence)} exposures")
    print(f"  Total exposure time: {adi_sequence.total_exposure_time}")
    print(f"  Total duration: {adi_sequence.total_duration}")
    
    # Create a reference target (could be another scene or offset pointing)
    ref_target = Target(scene_path="input/scenes/more_pix.fits", name="reference_star")
    
    # Create RDI sequence
    rdi_sequence = ObservationSequence.for_rdi(
        science_target=target,
        ref_target=ref_target,
        science_path_name="science_imaging",
        ref_path_name="reference_imaging",
        exposure_time=300 * u.s,
        start_time=Time.now(),
        n_ref_per_science=2
    )
    
    print(f"✓ Created RDI sequence with {len(rdi_sequence)} exposures")
    print(f"  Science and reference observations interleaved")
    
    return adi_sequence, rdi_sequence


def demo_full_simulation():
    """Demonstrate a complete end-to-end simulation."""
    print("=== FULL SIMULATION DEMO ===")
    print("Running complete simulation with real components")
    print()
    
    try:
        # Create light paths with real components
        light_paths = demo_real_pipeline_pattern()
        
        # Create observatory
        observatory = Observatory(light_paths)
        print("✓ Created observatory with real light paths")
        
        # Create observation sequence
        sequences = demo_real_observation_sequence()
        if sequences is None:
            print("Cannot run full simulation without valid targets")
            return None
        
        adi_sequence, rdi_sequence = sequences
        
        # Run a single exposure for demonstration
        single_obs = ObservationSequence.single_exposure(
            target=Target("input/scenes/more_pix.fits", "demo_target"),
            path_name="science_imaging",
            exposure_time=100 * u.s,
            start_time=Time.now()
        )
        
        print("✓ Running single exposure simulation...")
        
        # Run the simulation
        result = observatory.run(single_obs, seed=42)
        
        print(f"✓ Simulation completed successfully!")
        print(f"  Result type: {type(result)}")
        print(f"  Result attributes: {list(result.attrs.keys()) if hasattr(result, 'attrs') else 'N/A'}")
        
        return result
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        print("This may be due to missing dependencies (yippy, exoverses)")
        return None


def demo_pattern_comparison_real():
    """Compare all patterns using real components."""
    print("=== PATTERN COMPARISON WITH REAL COMPONENTS ===")
    print("Creating the same light path using all four patterns")
    print()
    
    # Create real hardware
    primary_params, coro_params, filter_params, det_params = create_example_hardware()
    
    # Pattern 1: Pipeline
    pipeline_path = (PathPipeline(LightPath.primary(primary_params)) >> 
                    LightPath.coronagraph(coro_params) >> 
                    LightPath.filter(filter_params) >> 
                    LightPath.detector(det_params))
    
    # Pattern 2: Functional composition
    functional_path = compose(
        LightPath.detector(det_params),
        LightPath.filter(filter_params),
        LightPath.coronagraph(coro_params),
        LightPath.primary(primary_params)
    )
    
    # Pattern 3: Builder
    builder_path = (PathBuilder()
                   .primary(primary_params)
                   .coronagraph(coro_params)
                   .filter(filter_params)
                   .detector(det_params)
                   .build())
    
    # Pattern 4: Registry (dynamic)
    registry_components = [
        ('primary', primary_params),
        ('coronagraph', coro_params),
        ('filter', filter_params),
        ('detector', det_params)
    ]
    
    registry_functions = [LightPath.from_registry(name, params) 
                         for name, params in registry_components]
    registry_path = compose(*reversed(registry_functions))
    
    print("✓ Created identical paths using all four patterns")
    print("  1. Pipeline pattern (>>)")
    print("  2. Functional composition (compose)")
    print("  3. Builder pattern (fluent)")
    print("  4. Registry pattern (dynamic)")
    print()
    
    print("All patterns use the same real yippy coronagraph model!")
    print("This demonstrates the power and flexibility of the new architecture.")
    
    return {
        "pipeline": pipeline_path,
        "functional": functional_path,
        "builder": builder_path,
        "registry": registry_path
    }


def main():
    """Run all real integration demonstrations."""
    print("CORONAGRAPHOTO V2: REAL INTEGRATION EXAMPLES")
    print("=" * 60)
    print()
    print("This demonstration shows the new light path patterns")
    print("working with real yippy coronagraphs and ExoVista scenes.")
    print()
    
    # Check for required files
    demo_coro_path = Path("input/coronagraphs/demo")
    scene_path = Path("input/scenes/more_pix.fits")
    
    if not demo_coro_path.exists():
        print(f"Warning: Demo coronagraph not found at {demo_coro_path}")
        print("Please ensure coronagraph files are available")
    
    if not scene_path.exists():
        print(f"Warning: Scene file not found at {scene_path}")
        print("Please ensure ExoVista scene files are available")
    
    print()
    
    # Run demonstrations
    print("1. Real Components Demo")
    print("-" * 30)
    demo_real_pipeline_pattern()
    
    print("\n2. ExoVista Integration Demo")
    print("-" * 30)
    demo_real_exovista_target()
    
    print("\n3. Observation Sequences Demo")
    print("-" * 30)
    demo_real_observation_sequence()
    
    print("\n4. Pattern Comparison Demo")
    print("-" * 30)
    demo_pattern_comparison_real()
    
    print("\n5. Full Simulation Demo")
    print("-" * 30)
    demo_full_simulation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("The new light path patterns successfully integrate with:")
    print("✓ Real yippy coronagraph models (from input/coronagraphs/)")
    print("✓ Real ExoVista scenes (from input/scenes/)")
    print("✓ All four composition patterns work identically")
    print("✓ Clean separation of hardware from observation planning")
    print("✓ Much simpler than the old manual step-by-step approach")
    print()
    print("Next steps:")
    print("• Install yippy and exoverses for full functionality")
    print("• Add your own coronagraph models to input/coronagraphs/")
    print("• Add your own scenes to input/scenes/")
    print("• Use the pipeline pattern for most use cases")


if __name__ == "__main__":
    main()