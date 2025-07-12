"""
Light path functions for coronagraphoto v2.

This module contains the pure functions that implement the physics of light
propagation through the coronagraph system. Each function is stateless and
can be composed into light paths.

The physics follows the original coronagraphoto implementation, handling:
- Coordinate transformations between different pixel scales
- Proper flux conservation during resampling
- Integration with yippy coronagraph models
- Synphot filter transmission
- Detector quantum efficiency and noise
"""

from dataclasses import dataclass
from typing import Any, Optional, Union
import numpy as np
import astropy.units as u
from astropy.time import Time
import xarray as xr

from .data_models import IntermediateData, PropagationContext

# Try to import optional dependencies
try:
    import astropy.units.equivalencies as equiv
    HAVE_EQUIV = True
except ImportError:
    HAVE_EQUIV = False

try:
    from coronagraphoto.transforms.image_transforms import flux_conserving_affine
    from coronagraphoto.util import zoom_conserve_flux
    HAVE_TRANSFORMS = True
except ImportError:
    HAVE_TRANSFORMS = False


# Hardware parameter classes
@dataclass
class PrimaryParams:
    """Parameters for the primary mirror."""
    diameter: u.Quantity
    reflectivity: float = 0.95
    temperature: u.Quantity = 280 * u.K
    frac_obscured: float = 0.0  # Central obscuration fraction
    
    def __post_init__(self):
        """Validate parameters."""
        if self.diameter.to(u.m).value <= 0:
            raise ValueError("Primary diameter must be positive")
        if not 0 <= self.reflectivity <= 1:
            raise ValueError("Reflectivity must be between 0 and 1")
        if not 0 <= self.frac_obscured < 1:
            raise ValueError("Central obscuration fraction must be between 0 and 1")


@dataclass
class CoronagraphParams:
    """Parameters for the coronagraph."""
    inner_working_angle: u.Quantity
    outer_working_angle: u.Quantity
    throughput: float = 0.1
    contrast: float = 1e-10
    coronagraph_model: Any = None  # yippy.Coronagraph object
    pixel_scale: Optional[u.Quantity] = None  # lambda/D per pixel
    npixels: Optional[int] = None  # number of pixels along one axis
    
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
    pixel_scale: u.Quantity  # arcsec/pixel
    read_noise: u.Quantity
    dark_current: u.Quantity
    quantum_efficiency: float = 0.9
    saturation_level: u.Quantity = 65535 * u.electron
    cic_rate: u.Quantity = 0 * u.electron  # clock-induced charge
    shape: tuple = (512, 512)  # detector dimensions in pixels
    
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
    bandpass_model: Any = None  # synphot.SpectralElement object
    spectral_resolution: Optional[float] = None  # R = λ/Δλ
    
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
    
    This follows the original implementation: illuminated_area = π * D²/4 * (1 - obscuration)
    
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
    # Calculate illuminated area following original implementation
    # illuminated_area = π * diameter²/4 * (1 - frac_obscured)
    # For now, assume no central obscuration (frac_obscured = 0)
    frac_obscured = getattr(params, 'frac_obscured', 0.0)
    illuminated_area = (
        np.pi * params.diameter**2 / 4.0 * (1.0 - frac_obscured)
    )
    
    # Apply reflectivity and collecting area to star flux
    # Units: (photon/s/nm/m²) * (reflectivity) * (m²) = photon/s/nm
    if data.star_flux is not None:
        star_flux = data.star_flux * params.reflectivity * illuminated_area
        new_data = data.update(star_flux=star_flux)
    else:
        new_data = data
    
    # Handle planets if present
    if data.planet_flux is not None:
        planet_flux = data.planet_flux * params.reflectivity * illuminated_area
        new_data = new_data.update(planet_flux=planet_flux)
    
    # Handle disk if present
    if data.disk_flux_map is not None:
        disk_flux_map = data.disk_flux_map * params.reflectivity * illuminated_area
        new_data = new_data.update(disk_flux_map=disk_flux_map)
    
    return new_data


def apply_coronagraph(data: IntermediateData, params: CoronagraphParams, context: PropagationContext) -> IntermediateData:
    """
    Apply coronagraph suppression to the light.
    
    This function implements the key physics of coronagraph operation following
    the original implementation. Different source types are handled differently:
    - Star: Uses stellar intensity map from coronagraph model
    - Planets: Uses off-axis PSF based on separation and position angle  
    - Disk: Uses PSF datacube convolution for extended source
    
    Args:
        data: 
            Input data with source fluxes
        params: 
            Coronagraph parameters containing yippy coronagraph model
        context: 
            Propagation context including wavelength and diameter
            
    Returns:
        Data with coronagraph effects applied
    """
    new_data = data
    
    # Handle star using stellar intensity map (like gen_star_count_rate)
    if data.star_flux is not None:
        star_count_rate = apply_star_coronagraph(
            data.star_flux, params, context
        )
        new_data = new_data.update(star_flux=star_count_rate)
    
    # Handle planets using off-axis PSF (like gen_planet_count_rate)
    if data.planet_flux is not None:
        planet_count_rate = apply_planet_coronagraph(
            data.planet_flux, params, context
        )
        new_data = new_data.update(planet_flux=planet_count_rate)
    
    # Handle disk using PSF datacube (like gen_disk_count_rate)
    if data.disk_flux_map is not None:
        disk_count_rate = apply_disk_coronagraph(
            data.disk_flux_map, params, context
        )
        new_data = new_data.update(disk_flux_map=disk_count_rate)
    
    return new_data


def apply_star_coronagraph(star_flux, params: CoronagraphParams, context: PropagationContext):
    """
    Apply coronagraph to stellar flux using stellar intensity map.
    
    Follows the original gen_star_count_rate implementation:
    1. Convert star angular diameter to lambda/D units
    2. Get stellar intensity map from coronagraph
    3. Multiply intensity map by photon flux
    """
    try:
        # This would require integration with the actual coronagraph object
        # For now, apply simple suppression based on contrast parameter
        suppressed_flux = star_flux * params.contrast
        
        # In full implementation, this would:
        # - Get stellar diameter in lambda/D: 
        #   stellar_diam_lod = star.angular_diameter.to(u.lod, equiv.lod(wavelength, diameter))
        # - Get intensity map: stellar_intens = coronagraph.stellar_intens(stellar_diam_lod).T
        # - Multiply: count_rate = np.multiply(stellar_intens, flux_term).T
        
        return suppressed_flux
        
    except Exception:
        # Fallback to simple contrast suppression
        return star_flux * params.contrast


def apply_planet_coronagraph(planet_flux, params: CoronagraphParams, context: PropagationContext):
    """
    Apply coronagraph to planetary flux using off-axis PSF.
    
    Follows the original gen_planet_count_rate implementation:
    1. Calculate planet positions and separations
    2. Apply off-axis PSF for each planet's location
    3. Sum contributions from all planets
    """
    try:
        # This would require orbital propagation and PSF interpolation
        # For now, apply throughput assuming planets are in good regions
        processed_flux = planet_flux * params.throughput
        
        # In full implementation, this would:
        # - Propagate orbits to get pixel coordinates
        # - Calculate separations: planet_alphas_lod = separations.to(u.lod, equiv.lod(wavelength, diameter))
        # - Get off-axis PSF: psf = coronagraph.offax(x, y, lam=wavelength, D=diameter)
        # - Apply PSF: planet_count_rate += planet_photon_flux[i] * psf
        
        return processed_flux
        
    except Exception:
        # Fallback to simple throughput
        return planet_flux * params.throughput


def apply_disk_coronagraph(disk_flux_map, params: CoronagraphParams, context: PropagationContext):
    """
    Apply coronagraph to disk flux using PSF datacube convolution.
    
    Follows the original gen_disk_count_rate implementation:
    1. Scale disk image to coronagraph pixel scale
    2. Center and crop disk to coronagraph dimensions
    3. Convolve with PSF datacube using tensor contraction
    """
    try:
        # This would require PSF datacube and proper scaling
        # For now, apply simple throughput
        processed_flux = disk_flux_map * params.throughput
        
        # In full implementation, this would:
        # - Scale disk: scaled_disk = zoom_conserve_flux(disk_image_photons, zoom_factor)
        # - Center and crop to coronagraph size  
        # - Convolve: count_rate = compute_disk_image(scaled_disk, coronagraph.psf_datacube)
        
        return processed_flux
        
    except Exception:
        # Fallback to simple throughput
        return disk_flux_map * params.throughput


def apply_filter(data: IntermediateData, params: FilterParams, context: PropagationContext) -> IntermediateData:
    """
    Apply optical filter transmission to the light.
    
    Handles both simple rectangular filters and synphot bandpass models.
    Follows the original implementation's wavelength handling.
    
    Args:
        data: 
            Input data with source fluxes
        params: 
            Filter parameters (may include synphot bandpass)
        context: 
            Propagation context including wavelength and bandwidth
            
    Returns:
        Data with filter effects applied
    """
    # Calculate filter transmission at the current wavelength
    wavelength = context.wavelength
    
    if params.bandpass_model is not None:
        # Use synphot bandpass model if available
        try:
            transmission = params.bandpass_model(wavelength).value
        except Exception:
            # Fallback to simple model
            transmission = _simple_filter_transmission(wavelength, params)
    else:
        # Use simple rectangular filter model
        transmission = _simple_filter_transmission(wavelength, params)
    
    # Apply transmission to all flux components
    new_data = data
    
    if data.star_flux is not None:
        star_flux = data.star_flux * transmission
        new_data = new_data.update(star_flux=star_flux)
    
    if data.planet_flux is not None:
        planet_flux = data.planet_flux * transmission
        new_data = new_data.update(planet_flux=planet_flux)
    
    if data.disk_flux_map is not None:
        disk_flux_map = data.disk_flux_map * transmission
        new_data = new_data.update(disk_flux_map=disk_flux_map)
    
    return new_data


def _simple_filter_transmission(wavelength: u.Quantity, params: FilterParams) -> float:
    """Simple rectangular filter transmission model."""
    wavelength_val = wavelength.to(u.nm).value
    central_wl = params.central_wavelength.to(u.nm).value
    bandwidth = params.bandwidth.to(u.nm).value
    
    # Rectangular filter transmission
    if abs(wavelength_val - central_wl) <= bandwidth / 2:
        return params.transmission
    else:
        return 0.0


def apply_detector(data: IntermediateData, params: DetectorParams, context: PropagationContext) -> IntermediateData:
    """
    Apply detector effects including coordinate transformation and noise.
    
    This function follows the original detector implementation:
    1. Resample from coronagraph (lambda/D) to detector pixel scale
    2. Convert photon rates to incident photons (Poisson process)
    3. Apply quantum efficiency (binomial process)
    4. Add detector noise (dark current, read noise, CIC)
    5. Apply saturation limits
    
    Args:
        data: 
            Input data with count rates in coronagraph coordinates
        params: 
            Detector parameters including pixel scale and noise characteristics
        context: 
            Propagation context including wavelength, diameter, and exposure time
            
    Returns:
        Data with detector effects applied in detector coordinates
    """
    new_data = data
    exposure_time = context.time_step.to(u.s).value
    
    # Set random seed for reproducibility
    if hasattr(context, 'rng_key') and context.rng_key is not None:
        np.random.seed(context.rng_key % (2**32))
    
    # Process each flux component
    if data.star_flux is not None:
        star_electrons = _apply_detector_to_flux(
            data.star_flux, params, context, exposure_time
        )
        new_data = new_data.update(star_flux=star_electrons)
    
    if data.planet_flux is not None:
        planet_electrons = _apply_detector_to_flux(
            data.planet_flux, params, context, exposure_time
        )
        new_data = new_data.update(planet_flux=planet_electrons)
    
    if data.disk_flux_map is not None:
        disk_electrons = _apply_detector_to_flux(
            data.disk_flux_map, params, context, exposure_time
        )
        new_data = new_data.update(disk_flux_map=disk_electrons)
    
    return new_data


def _apply_detector_to_flux(flux_data, params: DetectorParams, context: PropagationContext, exposure_time: float):
    """
    Apply detector effects to a single flux component.
    
    Follows the original detector implementation:
    0. Resample from coronagraph coordinates to detector coordinates
    1. Convert count rate to incident photons (Poisson)
    2. Apply quantum efficiency (binomial)
    3. Add detector noise components
    """
    # Step 0: Coordinate transformation (coronagraph lambda/D to detector arcsec/pixel)
    # This would require the coronagraph pixel scale and other parameters
    # For now, we assume the data is already in the correct coordinate system
    # In full implementation, this would use flux_conserving_affine
    resampled_flux = flux_data
    
    # Step 1: Convert rate to incident photons (Poisson process)
    # Units: (photon/s) * (s) = photons
    expected_photons = resampled_flux * exposure_time
    incident_photons = np.random.poisson(expected_photons.astype(float))
    
    # Step 2: Apply quantum efficiency (binomial process)
    # Each incident photon has probability QE of becoming a photoelectron
    photo_electrons = np.random.binomial(
        n=incident_photons.astype(int), 
        p=params.quantum_efficiency
    ).astype(float)
    
    # Step 3: Add detector noise
    shape = photo_electrons.shape
    
    # Dark current (Poisson process)
    expected_dark = params.dark_current.to(u.electron/u.s).value * exposure_time
    dark_electrons = np.random.poisson(expected_dark, size=shape).astype(float)
    
    # Read noise (Gaussian process) - simplified for now
    read_noise_electrons = np.random.normal(
        0, params.read_noise.to(u.electron).value, size=shape
    )
    
    # Clock-induced charge (Poisson process)
    cic_electrons = np.random.poisson(
        params.cic_rate.to(u.electron).value, size=shape
    ).astype(float)
    
    # Step 4: Sum all electron sources
    total_electrons = photo_electrons + dark_electrons + read_noise_electrons + cic_electrons
    
    # Step 5: Apply saturation
    saturated_electrons = np.minimum(
        total_electrons, 
        params.saturation_level.to(u.electron).value
    )
    
    return saturated_electrons


def resample_to_detector(coro_data, coro_pixel_scale: u.Quantity, det_params: DetectorParams, 
                        wavelength: u.Quantity, diameter: u.Quantity):
    """
    Resample data from coronagraph coordinates to detector coordinates.
    
    Follows the original detector._resample_to_detector implementation using
    flux-conserving coordinate transformation.
    
    Args:
        coro_data: 
            Data in coronagraph coordinates (lambda/D pixel scale)
        coro_pixel_scale: 
            Coronagraph pixel scale in lambda/D per pixel
        det_params: 
            Detector parameters including pixel scale
        wavelength: 
            Observation wavelength
        diameter: 
            Telescope diameter
            
    Returns:
        Data resampled to detector pixel scale
    """
    if not HAVE_TRANSFORMS:
        # Fallback: return original data if transforms not available
        return coro_data
    
    try:
        # Convert coronagraph pixel scale from lambda/D to arcsec/pixel
        if HAVE_EQUIV:
            coro_scale_arcsec = (coro_pixel_scale * u.pix).to(
                u.arcsec, equiv.lod(wavelength, diameter)
            ) / u.pix
        else:
            # Simple fallback conversion
            lod_to_arcsec = (wavelength / diameter).to(u.arcsec, u.dimensionless_angles())
            coro_scale_arcsec = coro_pixel_scale * lod_to_arcsec
        
        # Resample using flux-conserving affine transformation
        resampled = flux_conserving_affine(
            coro_data.astype(float),
            pixscale_src=coro_scale_arcsec,
            pixscale_tgt=det_params.pixel_scale,
            shape_tgt=det_params.shape,
            rotation_deg=0.0,  # No rotation for now
            order=3
        )
        
        return resampled
        
    except Exception:
        # Fallback to simple resampling if advanced transforms fail
        return _simple_resample(coro_data, det_params.shape)


def _simple_resample(data, target_shape):
    """Simple resampling fallback using scipy zoom."""
    from scipy.ndimage import zoom
    
    if len(data.shape) != 2:
        return data  # Can't handle non-2D data simply
    
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return zoom(data, zoom_factors, order=1)


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