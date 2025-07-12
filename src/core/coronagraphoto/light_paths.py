"""
Light path functions for coronagraphoto v2.

This module contains the pure functions that implement the physics of light
propagation through the coronagraph system. Each function is stateless and
can be composed into light paths.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import astropy.units as u
from astropy.time import Time
import xarray as xr

from .data_models import IntermediateData, PropagationContext


# Hardware parameter classes
@dataclass
class PrimaryParams:
    """Parameters for the primary mirror."""
    diameter: u.Quantity
    reflectivity: float = 0.95
    temperature: u.Quantity = 280 * u.K
    
    def __post_init__(self):
        """Validate parameters."""
        if self.diameter.to(u.m).value <= 0:
            raise ValueError("Primary diameter must be positive")
        if not 0 <= self.reflectivity <= 1:
            raise ValueError("Reflectivity must be between 0 and 1")


@dataclass
class CoronagraphParams:
    """Parameters for the coronagraph."""
    inner_working_angle: u.Quantity
    outer_working_angle: u.Quantity
    throughput: float = 0.1
    contrast: float = 1e-10
    
    def __post_init__(self):
        """Validate parameters."""
        if self.inner_working_angle.to(u.arcsec).value <= 0:
            raise ValueError("Inner working angle must be positive")
        if self.outer_working_angle <= self.inner_working_angle:
            raise ValueError("Outer working angle must be greater than inner working angle")
        if not 0 <= self.throughput <= 1:
            raise ValueError("Throughput must be between 0 and 1")
        if self.contrast <= 0:
            raise ValueError("Contrast must be positive")


@dataclass
class DetectorParams:
    """Parameters for the detector."""
    pixel_scale: u.Quantity
    read_noise: u.Quantity
    dark_current: u.Quantity
    quantum_efficiency: float = 0.9
    saturation_level: u.Quantity = 65535 * u.adu
    
    def __post_init__(self):
        """Validate parameters."""
        if self.pixel_scale.to(u.arcsec).value <= 0:
            raise ValueError("Pixel scale must be positive")
        if self.read_noise.to(u.electron).value < 0:
            raise ValueError("Read noise must be non-negative")
        if self.dark_current.to(u.electron/u.s).value < 0:
            raise ValueError("Dark current must be non-negative")
        if not 0 <= self.quantum_efficiency <= 1:
            raise ValueError("Quantum efficiency must be between 0 and 1")


@dataclass
class FilterParams:
    """Parameters for optical filters."""
    central_wavelength: u.Quantity
    bandwidth: u.Quantity
    transmission: float = 0.8
    
    def __post_init__(self):
        """Validate parameters."""
        if self.central_wavelength.to(u.nm).value <= 0:
            raise ValueError("Central wavelength must be positive")
        if self.bandwidth.to(u.nm).value <= 0:
            raise ValueError("Bandwidth must be positive")
        if not 0 <= self.transmission <= 1:
            raise ValueError("Transmission must be between 0 and 1")


# Light path functions
def apply_primary(data: IntermediateData, params: PrimaryParams, context: PropagationContext) -> IntermediateData:
    """
    Apply primary mirror effects to the light.
    
    Args:
        data: 
            Input data with source fluxes
        params: 
            Primary mirror parameters
        context: 
            Propagation context
            
    Returns:
        Data with primary mirror effects applied
    """
    # Calculate collecting area
    collecting_area = np.pi * (params.diameter / 2) ** 2
    
    # Apply reflectivity and collecting area
    star_flux = data.star_flux * params.reflectivity * collecting_area.to(u.m**2).value
    
    # Update the data
    new_data = data.update(star_flux=star_flux)
    
    # Handle planets if present
    if data.planet_flux is not None:
        planet_flux = data.planet_flux * params.reflectivity * collecting_area.to(u.m**2).value
        new_data = new_data.update(planet_flux=planet_flux)
    
    return new_data


def apply_coronagraph(data: IntermediateData, params: CoronagraphParams, context: PropagationContext) -> IntermediateData:
    """
    Apply coronagraph suppression to the light.
    
    This function implements the key physics of coronagraph operation:
    different suppression for star vs planets based on angular separation.
    
    Args:
        data: 
            Input data with source fluxes
        params: 
            Coronagraph parameters
        context: 
            Propagation context
            
    Returns:
        Data with coronagraph effects applied
    """
    # Suppress stellar light by the coronagraph contrast
    suppressed_star_flux = data.star_flux * params.contrast
    
    # For planets, suppression depends on angular separation
    # This is a simplified model - in practice this would be more complex
    new_data = data.update(star_flux=suppressed_star_flux)
    
    if data.planet_flux is not None:
        # Assume planets are at separations where coronagraph is effective
        # Apply throughput but not full suppression
        planet_flux = data.planet_flux * params.throughput
        new_data = new_data.update(planet_flux=planet_flux)
    
    return new_data


def apply_filter(data: IntermediateData, params: FilterParams, context: PropagationContext) -> IntermediateData:
    """
    Apply optical filter transmission to the light.
    
    Args:
        data: 
            Input data with source fluxes
        params: 
            Filter parameters
        context: 
            Propagation context
            
    Returns:
        Data with filter effects applied
    """
    # Calculate filter transmission at the current wavelength
    # Simple rectangular filter model
    wavelength = context.wavelength.to(u.nm).value
    central_wl = params.central_wavelength.to(u.nm).value
    bandwidth = params.bandwidth.to(u.nm).value
    
    # Rectangular filter transmission
    if abs(wavelength - central_wl) <= bandwidth / 2:
        transmission = params.transmission
    else:
        transmission = 0.0
    
    # Apply transmission
    star_flux = data.star_flux * transmission
    new_data = data.update(star_flux=star_flux)
    
    if data.planet_flux is not None:
        planet_flux = data.planet_flux * transmission
        new_data = new_data.update(planet_flux=planet_flux)
    
    return new_data


def apply_detector(data: IntermediateData, params: DetectorParams, context: PropagationContext) -> IntermediateData:
    """
    Apply detector effects including noise and digitization.
    
    Args:
        data: 
            Input data with source fluxes
        params: 
            Detector parameters
        context: 
            Propagation context
            
    Returns:
        Data with detector effects applied
    """
    # Convert flux to electrons
    # This is a simplified model - in practice would depend on exact units
    exposure_time = context.time_step.to(u.s).value
    
    # Apply quantum efficiency
    star_electrons = data.star_flux * params.quantum_efficiency * exposure_time
    
    # Add detector noise
    # Shot noise (Poisson)
    if hasattr(context, 'rng_key') and context.rng_key is not None:
        np.random.seed(context.rng_key % (2**32))
    
    # Simplified noise model - in practice would be more sophisticated
    read_noise_electrons = np.random.normal(0, params.read_noise.to(u.electron).value, 
                                           size=star_electrons.shape)
    dark_noise_electrons = np.random.poisson(
        params.dark_current.to(u.electron/u.s).value * exposure_time,
        size=star_electrons.shape
    )
    
    # Total signal including noise
    total_signal = star_electrons + read_noise_electrons + dark_noise_electrons
    
    # Apply saturation
    total_signal = np.minimum(total_signal, params.saturation_level.to(u.electron).value)
    
    # Convert to ADU (assuming 1 electron = 1 ADU for simplicity)
    final_signal = total_signal
    
    # Update the data
    new_data = data.update(star_flux=final_signal)
    
    # Handle planets if present
    if data.planet_flux is not None:
        planet_electrons = data.planet_flux * params.quantum_efficiency * exposure_time
        planet_signal = planet_electrons + read_noise_electrons + dark_noise_electrons
        planet_signal = np.minimum(planet_signal, params.saturation_level.to(u.electron).value)
        new_data = new_data.update(planet_flux=planet_signal)
    
    return new_data


def apply_speckles(data: IntermediateData, params: Any, context: PropagationContext) -> IntermediateData:
    """
    Apply speckle noise to the data.
    
    Args:
        data: 
            Input data
        params: 
            Speckle parameters (placeholder)
        context: 
            Propagation context
            
    Returns:
        Data with speckle effects applied
    """
    # This is a placeholder for speckle modeling
    # In practice, this would add coherent speckle noise
    
    # For now, just add some correlated noise
    if hasattr(context, 'rng_key') and context.rng_key is not None:
        np.random.seed(context.rng_key % (2**32))
    
    # Add simple speckle-like noise to star flux
    speckle_noise = np.random.normal(0, 0.01 * data.star_flux.values, 
                                    size=data.star_flux.shape)
    
    noisy_star_flux = data.star_flux + speckle_noise
    
    return data.update(star_flux=noisy_star_flux)


def load_scene(target_path: str, context: PropagationContext) -> IntermediateData:
    """
    Load a scene from a file into IntermediateData format.
    
    Args:
        target_path: 
            Path to the scene file
        context: 
            Propagation context
            
    Returns:
        IntermediateData loaded from the scene file
    """
    # This is a placeholder implementation
    # In practice, this would load from ExoVista or other scene formats
    
    # Create a simple stellar spectrum for testing
    wavelengths = np.linspace(400, 800, 100) * u.nm
    
    # Simple blackbody-like spectrum
    flux_values = np.exp(-(wavelengths.value - 550)**2 / (2 * 50**2))
    flux = flux_values * 1e6  # Arbitrary units
    
    # Create the IntermediateData
    dataset = xr.Dataset({
        'star_flux': xr.DataArray(
            flux,
            dims=['wavelength'],
            coords={'wavelength': wavelengths.value},
            attrs={'units': 'photons/s/nm'}
        )
    })
    
    # Add scene metadata
    dataset.attrs.update({
        'scene_file': target_path,
        'loaded_at': context.time.iso if context.time else 'unknown'
    })
    
    return IntermediateData(dataset)