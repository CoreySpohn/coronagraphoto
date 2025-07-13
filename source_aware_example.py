#!/usr/bin/env python3
"""
Example of the source-aware light path system.

This shows how a user defines a single pipeline with the >> operator,
but internally it creates three different paths for star, planets, and disk
since they interact differently with the coronagraph.
"""

import sys
import numpy as np
import astropy.units as u
from astropy.time import Time

sys.path.insert(0, 'src/core')

from coronagraphoto.light_paths import (
    Path, PathPipeline,
    PrimaryParams, CoronagraphParams, FilterParams, DetectorParams
)
from coronagraphoto.source_aware_paths import (
    SourceAwareLightPath, create_coronagraph_aware_path
)
from coronagraphoto.data_models import PropagationContext


def main():
    """Demonstrate the source-aware pipeline system."""
    print("🌟 SOURCE-AWARE LIGHT PATH DEMONSTRATION")
    print("=" * 50)
    print()
    
    # Step 1: User defines hardware parameters
    print("📋 User defines hardware parameters...")
    
    primary_params = PrimaryParams(diameter=6.5 * u.m)
    coronagraph_params = CoronagraphParams(coronagraph_dir="input/coronagraphs/demo")
    filter_params = FilterParams(
        central_wavelength=550 * u.nm,
        bandwidth=20 * u.nm
    )
    detector_params = DetectorParams(
        pixel_scale=0.01 * u.arcsec,
        read_noise=3 * u.electron,
        dark_current=0.001 * u.electron / u.s,
        shape=(256, 256)
    )
    print("✅ Hardware parameters defined")
    print()
    
    # Step 2: User creates a single pipeline using >> operator
    print("🔗 User creates a single pipeline...")
    
    user_pipeline = (PathPipeline(Path.primary(primary_params)) >> 
                    Path.coronagraph(coronagraph_params) >> 
                    Path.filter(filter_params) >> 
                    Path.detector(detector_params))
    
    print("✅ Pipeline: Primary >> Coronagraph >> Filter >> Detector")
    print()
    
    # Step 3: System internally creates source-aware paths
    print("⚙️  System internally creates source-aware paths...")
    
    # Convert to source-aware path
    # This happens automatically in the observatory
    aware_path = create_coronagraph_aware_path(
        [Path.primary(primary_params),
         Path.coronagraph(coronagraph_params),
         Path.filter(filter_params),
         Path.detector(detector_params)],
        coronagraph_params
    )
    
    print("✅ Three internal paths created:")
    print("   - Star path: Uses stellar intensity map")
    print("   - Planet path: Uses off-axis PSFs") 
    print("   - Disk path: Uses PSF datacube convolution")
    print()
    
    # Step 4: When executed, each source is processed appropriately
    print("🚀 During execution, each source is processed differently...")
    print()
    
    # Create a mock ExoVista system
    class MockSystem:
        class Star:
            def spec_flux_density(self, wavelength, time):
                return 1e-10 * u.Jy
                
        class Planet:
            def spec_flux_density(self, wavelength, time):
                return 1e-12 * u.Jy
                
        class Disk:
            def spec_flux_density(self, wavelength, time):
                # Return a 2D flux map
                return np.ones((128, 128)) * 1e-13 * u.Jy
                
        def __init__(self):
            self.star = self.Star()
            self.planets = [self.Planet(), self.Planet()]
            self.disk = self.Disk()
    
    system = MockSystem()
    
    # Create propagation context
    context = PropagationContext(
        time=Time.now(),
        wavelength=550 * u.nm,
        bandpass_slice=1.0 * u.nm,
        time_step=100 * u.s,
        rng_key=42
    )
    
    # Execute the path (this would happen in the observatory)
    print("📡 Processing star...")
    print("   - Evaluates star.spec_flux_density(550nm, now)")
    print("   - Applies stellar intensity map from coronagraph")
    print("   - Result: 2D stellar flux map")
    print()
    
    print("🪐 Processing planets...")
    print("   - Evaluates each planet.spec_flux_density(550nm, now)")
    print("   - Propagates orbits to get positions")
    print("   - Applies off-axis PSF for each planet position")
    print("   - Result: Sum of planet PSF maps")
    print()
    
    print("💿 Processing disk...")
    print("   - Evaluates disk.spec_flux_density(550nm, now)")
    print("   - Resamples disk to coronagraph pixel scale using flux_conserving_affine")
    print("   - Convolves with PSF datacube")
    print("   - Result: Convolved disk map")
    print()
    
    # Step 5: User can control which sources to include
    print("🎛️  User can control which sources to include...")
    
    # Create path with only star and planets
    aware_path_subset = SourceAwareLightPath(
        user_path=[],  # Would be the actual path functions
        include_sources={'star': True, 'planets': True, 'disk': False}
    )
    
    print("✅ Created path with star and planets only (disk disabled)")
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 50)
    print("✅ User defines ONE pipeline with >> operator")
    print("✅ System creates THREE internal paths automatically") 
    print("✅ Each source type uses correct coronagraph physics:")
    print("   - Star → Stellar intensity map")
    print("   - Planets → Off-axis PSFs")
    print("   - Disk → PSF datacube convolution")
    print("✅ Flux evaluation happens at execution time")
    print("✅ Uses flux_conserving_affine for resampling")
    print("✅ User can enable/disable sources as needed")
    print()
    print("🎉 Clean, physics-aware architecture!")


if __name__ == "__main__":
    main()