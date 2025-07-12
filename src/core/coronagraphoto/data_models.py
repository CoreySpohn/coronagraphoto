"""
Data models for the coronagraphoto v2 architecture.

This module defines the core data structures used throughout the simulation:
- IntermediateData: Multi-component container for simulation data
- PropagationContext: State for a single propagation step
"""

from dataclasses import dataclass
from typing import Any, Union
import numpy as np
import astropy.units as u
from astropy.time import Time
import xarray as xr


@dataclass(frozen=True)
class PropagationContext:
    """
    Provides the state for a single propagation through the LightPath.
    
    This immutable context object contains all the time- and wavelength-dependent
    information needed for a single step in the light path simulation.
    """
    time: Time
    wavelength: u.Quantity
    bandpass_slice: u.Quantity  # The width of the wavelength bin, d(lambda)
    time_step: u.Quantity      # The duration of the time step, dt
    rng_key: Any               # JAX PRNG key or numpy random state


class IntermediateData:
    """
    A multi-component container for simulation data.
    
    This class wraps an xarray.Dataset that holds separate DataArrays for each
    source type (star, planets, disk). This separation is critical for proper
    physical modeling as different source types require different propagation
    methods through the coronagraph.
    
    A typical IntermediateData dataset contains:
    - star_flux: (wavelength,) - stellar spectral flux density
    - planet_flux: (planet, wavelength) - planetary spectral flux density
    - planet_coords: (planet, time, coord_xy) - planet sky coordinates
    - disk_flux_map: (x, y, wavelength) - disk spectral flux density map
    """
    
    def __init__(self, dataset: xr.Dataset):
        """
        Initialize with an xarray Dataset.
        
        Args:
            dataset: 
                xarray Dataset containing the separated source components
        """
        self._dataset = dataset
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate that the dataset has the expected structure."""
        required_vars = ['star_flux']
        for var in required_vars:
            if var not in self._dataset.data_vars:
                raise ValueError(f"Required variable '{var}' not found in dataset")
    
    @property
    def dataset(self) -> xr.Dataset:
        """Access the underlying xarray Dataset."""
        return self._dataset
    
    @property
    def star_flux(self) -> xr.DataArray:
        """Access the stellar flux data."""
        return self._dataset['star_flux']
    
    @property
    def planet_flux(self) -> Union[xr.DataArray, None]:
        """Access the planetary flux data if present."""
        return self._dataset.get('planet_flux')
    
    @property
    def planet_coords(self) -> Union[xr.DataArray, None]:
        """Access the planetary coordinates if present."""
        return self._dataset.get('planet_coords')
    
    @property
    def disk_flux_map(self) -> Union[xr.DataArray, None]:
        """Access the disk flux map if present."""
        return self._dataset.get('disk_flux_map')
    
    def copy(self) -> 'IntermediateData':
        """Create a copy of this IntermediateData object."""
        return IntermediateData(self._dataset.copy())
    
    def update(self, **kwargs) -> 'IntermediateData':
        """
        Create a new IntermediateData with updated variables.
        
        Args:
            **kwargs: New variables to add or update in the dataset
            
        Returns:
            New IntermediateData instance with updated data
        """
        new_dataset = self._dataset.copy()
        for name, data in kwargs.items():
            new_dataset[name] = data
        return IntermediateData(new_dataset)
    
    @classmethod
    def from_star_spectrum(cls, wavelength: u.Quantity, flux: u.Quantity) -> 'IntermediateData':
        """
        Create IntermediateData from a stellar spectrum.
        
        Args:
            wavelength: 
                Wavelength array with units
            flux: 
                Spectral flux density array with units
                
        Returns:
            IntermediateData instance with stellar spectrum
        """
        dataset = xr.Dataset({
            'star_flux': xr.DataArray(
                flux.value,
                dims=['wavelength'],
                coords={'wavelength': wavelength.value},
                attrs={'units': str(flux.unit)}
            )
        })
        return cls(dataset)
    
    @classmethod
    def from_scene_file(cls, scene_path: str) -> 'IntermediateData':
        """
        Create IntermediateData from a scene file.
        
        Args:
            scene_path: 
                Path to the scene file (e.g., ExoVista FITS file)
                
        Returns:
            IntermediateData instance loaded from the scene
        """
        # For now, import the scene loading function
        from .light_paths import load_scene
        from astropy.time import Time
        
        # Create a dummy context for loading
        context = PropagationContext(
            time=Time.now(),
            wavelength=550 * u.nm,
            bandpass_slice=10 * u.nm,
            time_step=1 * u.s,
            rng_key=42
        )
        
        return load_scene(scene_path, context)