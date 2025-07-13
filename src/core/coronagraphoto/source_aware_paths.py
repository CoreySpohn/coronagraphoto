"""
Source-aware light path system for coronagraphoto v2.

This module implements the proper physics where star, planets, and disk
interact differently with the coronagraph:
- Star: Uses stellar intensity map  
- Planets: Use off-axis PSFs based on position
- Disk: Uses PSF datacube convolution

The user defines a single light path, but internally we apply it differently
to each source type when the coronagraph step is involved.
"""

from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
import numpy as np
import astropy.units as u
from astropy.time import Time
import xarray as xr

# Import the transform utilities
try:
    from ..transforms.image_transforms import flux_conserving_affine
    HAVE_TRANSFORMS = True
except ImportError:
    HAVE_TRANSFORMS = False
    
from .data_models import IntermediateData, PropagationContext


@dataclass 
class SourceData:
    """Container for data from a specific source type."""
    star_flux: Optional[Any] = None  # Scalar flux value
    planet_fluxes: Optional[List[Any]] = None  # List of planet fluxes
    planet_positions: Optional[List[Any]] = None  # List of (x,y) positions
    disk_flux_map: Optional[Any] = None  # 2D flux map
    
    
class SourceAwareLightPath:
    """
    A light path that handles different source types appropriately.
    
    This class takes a user-defined light path and creates three internal
    paths that handle star, planets, and disk differently when they
    encounter the coronagraph step.
    """
    
    def __init__(self, user_path: List[Callable], include_sources: Optional[Dict[str, bool]] = None):
        """
        Initialize with a user-defined path.
        
        Args:
            user_path: List of functions that process light
            include_sources: Dict controlling which sources to include
                           {'star': True, 'planets': True, 'disk': True}
        """
        self.user_path = user_path
        self.include_sources = include_sources or {
            'star': True,
            'planets': True, 
            'disk': True
        }
        
        # Create specialized paths for each source type
        self._create_source_specific_paths()
        
    def _create_source_specific_paths(self):
        """Create the three internal paths for star, planets, and disk."""
        # For now, just copy the user path
        # In a full implementation, we'd modify the coronagraph step
        # to use the appropriate interaction for each source
        self.star_path = self.user_path.copy()
        self.planet_path = self.user_path.copy()
        self.disk_path = self.user_path.copy()
        
    def __call__(self, system, context: PropagationContext) -> IntermediateData:
        """
        Execute the light path on an ExoVista system.
        
        Args:
            system: ExoVista system object with star, planets, disk
            context: Propagation context with time, wavelength, etc.
            
        Returns:
            IntermediateData with results from all enabled sources
        """
        # Extract time and wavelength from context
        time = context.time
        wavelength = context.wavelength
        
        results = {}
        
        # Process star
        if self.include_sources['star'] and hasattr(system, 'star'):
            # Evaluate star flux at this time/wavelength
            star_flux = system.star.spec_flux_density(wavelength, time)
            star_data = SourceData(star_flux=star_flux)
            
            # Run through star-specific path
            star_result = self._run_path(star_data, self.star_path, context, 'star')
            results['star_flux'] = star_result
            
        # Process planets  
        if self.include_sources['planets'] and hasattr(system, 'planets'):
            # Evaluate planet fluxes and positions
            planet_fluxes = []
            planet_positions = []
            
            for planet in system.planets:
                flux = planet.spec_flux_density(wavelength, time)
                planet_fluxes.append(flux)
                # Position would come from orbit propagation
                # For now, use placeholder
                planet_positions.append((0, 0))
                
            planet_data = SourceData(
                planet_fluxes=planet_fluxes,
                planet_positions=planet_positions
            )
            
            # Run through planet-specific path
            planet_result = self._run_path(planet_data, self.planet_path, context, 'planets')
            results['planet_flux'] = planet_result
            
        # Process disk
        if self.include_sources['disk'] and hasattr(system, 'disk'):
            # Evaluate disk flux map at this time/wavelength
            disk_flux_map = system.disk.spec_flux_density(wavelength, time)
            disk_data = SourceData(disk_flux_map=disk_flux_map)
            
            # Run through disk-specific path  
            disk_result = self._run_path(disk_data, self.disk_path, context, 'disk')
            results['disk_flux_map'] = disk_result
            
        # Combine results into IntermediateData
        dataset = xr.Dataset(results)
        return IntermediateData(dataset)
        
    def _run_path(self, source_data: SourceData, path: List[Callable], 
                  context: PropagationContext, source_type: str):
        """
        Run a source through its specific light path.
        
        Args:
            source_data: Data for this source type
            path: List of functions to apply
            context: Propagation context
            source_type: 'star', 'planets', or 'disk'
            
        Returns:
            Processed data after running through the path
        """
        # Start with the source data
        current_data = source_data
        
        # Apply each step in the path
        for step in path:
            # Each step should handle the specific source type appropriately
            current_data = step(current_data, context)
            
        return current_data


def create_coronagraph_aware_path(base_path: List[Callable], 
                                 coronagraph_params) -> SourceAwareLightPath:
    """
    Create a source-aware path from a base path.
    
    This function takes a user-defined path and creates a SourceAwareLightPath
    that handles the coronagraph step differently for each source type.
    
    Args:
        base_path: User-defined light path 
        coronagraph_params: Parameters for the coronagraph
        
    Returns:
        SourceAwareLightPath that handles sources appropriately
    """
    # Find the coronagraph step in the path
    modified_paths = {
        'star': [],
        'planets': [],
        'disk': []
    }
    
    for step in base_path:
        # Check if this is a coronagraph step
        if hasattr(step, '__name__') and 'coronagraph' in step.__name__.lower():
            # Replace with source-specific versions
            modified_paths['star'].append(
                lambda data, ctx: apply_coronagraph_to_star(data, ctx, coronagraph_params)
            )
            modified_paths['planets'].append(
                lambda data, ctx: apply_coronagraph_to_planets(data, ctx, coronagraph_params)
            )
            modified_paths['disk'].append(
                lambda data, ctx: apply_coronagraph_to_disk(data, ctx, coronagraph_params)
            )
        else:
            # Use the same step for all sources
            modified_paths['star'].append(step)
            modified_paths['planets'].append(step)
            modified_paths['disk'].append(step)
            
    # Create the source-aware path
    aware_path = SourceAwareLightPath(base_path)
    aware_path.star_path = modified_paths['star']
    aware_path.planet_path = modified_paths['planets']
    aware_path.disk_path = modified_paths['disk']
    
    return aware_path


def apply_coronagraph_to_star(data: SourceData, context: PropagationContext, 
                             coronagraph_params) -> SourceData:
    """Apply coronagraph stellar intensity map to star."""
    if data.star_flux is None:
        return data
        
    # Get the coronagraph model
    coronagraph = coronagraph_params.get_coronagraph_model()
    
    # Get stellar diameter in lambda/D units
    # This would use the actual star properties
    stellar_diam_lod = 0.01  # Placeholder
    
    # Get stellar intensity map
    stellar_intens = coronagraph.stellar_intens(stellar_diam_lod)
    
    # Apply to flux (this creates a 2D map from scalar flux)
    flux_map = data.star_flux * stellar_intens
    
    # Return updated data
    return SourceData(star_flux=flux_map)


def apply_coronagraph_to_planets(data: SourceData, context: PropagationContext,
                                coronagraph_params) -> SourceData:
    """Apply coronagraph off-axis PSFs to planets."""
    if data.planet_fluxes is None:
        return data
        
    # Get the coronagraph model
    coronagraph = coronagraph_params.get_coronagraph_model()
    
    # Process each planet
    planet_maps = []
    if data.planet_positions:
        for flux, (x, y) in zip(data.planet_fluxes, data.planet_positions):
            # Get off-axis PSF for this position
            # Note: diameter should come from observation scenario
            diameter = getattr(context, 'diameter', 6.5 * u.m)
            psf = coronagraph.offax(x, y, lam=context.wavelength, D=diameter)
            
            # Apply PSF to flux
            planet_map = flux * psf
            planet_maps.append(planet_map)
        
    # Sum all planet contributions
    if planet_maps:
        # Return a 2D flux map that's the sum of all planet PSFs
        total_planet_map = sum(planet_maps)
        return SourceData(disk_flux_map=total_planet_map)  # Store as map
    else:
        return data


def apply_coronagraph_to_disk(data: SourceData, context: PropagationContext,
                             coronagraph_params) -> SourceData:
    """Apply coronagraph PSF datacube to disk."""
    if data.disk_flux_map is None:
        return data
        
    # Get the coronagraph model
    coronagraph = coronagraph_params.get_coronagraph_model()
    
    # Make sure PSF datacube exists
    if not coronagraph.has_psf_datacube:
        coronagraph.create_psf_datacube()
        
    # First, we need to resample the disk to coronagraph pixel scale
    if HAVE_TRANSFORMS:
        # Get pixel scales
        disk_pixscale = 1.0  # Would come from disk metadata
        coro_pixscale = coronagraph.pixel_scale.value
        
        # Resample disk to coronagraph grid
        resampled_disk = flux_conserving_affine(
            data.disk_flux_map,
            disk_pixscale,
            coro_pixscale,
            shape_tgt=(coronagraph.npixels, coronagraph.npixels),
            rotation_deg=0.0
        )
    else:
        resampled_disk = data.disk_flux_map
        
    # Convolve with PSF datacube
    # This is the tensor contraction from observation.py
    convolved = np.einsum('ij,ijxy->xy', resampled_disk, coronagraph.psf_datacube)
    
    return SourceData(disk_flux_map=convolved)