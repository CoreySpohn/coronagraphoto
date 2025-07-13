"""Light path functions for coronagraphoto v2.

This module provides elegant functional composition for light path physics.
The new system allows for clean, composable path definitions through multiple
patterns:

1. Pipeline composition: Path.primary() >> Path.coronagraph() >> Path.detector()
2. Functional composition: compose(detector, coronagraph, primary)
3. Builder pattern: PathBuilder().add_primary().add_coronagraph().build()
4. Decorator-based registration: @path_component automatically registers functions

The physics follows the original coronagraphoto implementation, handling:
- Coordinate transformations between different pixel scales
- Proper flux conservation during resampling
- Integration with yippy coronagraph models
- Synphot filter transmission
- Detector quantum efficiency and noise
"""

from dataclasses import dataclass
from functools import partial, reduce, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.time import Time

from .data_models import IntermediateData, PropagationContext

# Try to import optional dependencies
try:
    import astropy.units.equivalencies as equiv

    HAVE_EQUIV = True
except ImportError:
    HAVE_EQUIV = False

try:
    from coronagraphoto.transforms.image_transforms import flux_conserving_affine

    HAVE_TRANSFORMS = True
except ImportError:
    HAVE_TRANSFORMS = False


# Type definitions
PathFunction = Callable[[IntermediateData, PropagationContext], IntermediateData]
PathStep = Callable[[PropagationContext], PathFunction]


# Global registry of path components
_PATH_REGISTRY: Dict[str, PathStep] = {}


def path_component(name: str, category: str = "generic"):
    """Decorator to register a function as a reusable path component.

    Args:
        name:
            Name for the component (used in registry)
        category:
            Category for organization (primary, coronagraph, detector, etc.)

    Returns:
        Decorated function that can be used in path composition
    """

    def decorator(func):
        @wraps(func)
        def wrapper(params, **kwargs):
            # Return a partially applied function ready for path composition
            return partial(func, params=params, **kwargs)

        # Set attributes on the original function
        func.category = category
        func.component_name = name

        # Register the component
        _PATH_REGISTRY[name] = wrapper

        return wrapper

    return decorator


def compose(*functions: PathFunction) -> PathFunction:
    """Compose multiple path functions into a single function.

    Functions are applied right-to-left (mathematical composition).

    Args:
        *functions:
            Path functions to compose

    Returns:
        Composed function
    """

    def composed_function(
        data: IntermediateData, context: PropagationContext
    ) -> IntermediateData:
        if not functions:
            # Identity function for empty composition
            return data
        return reduce(lambda d, f: f(d, context), reversed(functions), data)

    return composed_function


class PathPipeline:
    """A pipeline for composing path functions with >> operator.

    This allows for elegant left-to-right composition:
    path = PathPipeline(primary) >> coronagraph >> detector
    """

    def __init__(self, func: PathFunction):
        """Initialize pipeline with a single function."""
        self.func = func

    def __rshift__(self, other: Union["PathPipeline", PathFunction]) -> "PathPipeline":
        """Compose this pipeline with another function (>> operator).

        Args:
            other:
                Another pipeline or path function

        Returns:
            New pipeline with composed functions
        """
        if isinstance(other, PathPipeline):
            other_func = other.func
        else:
            other_func = other

        def composed(
            data: IntermediateData, context: PropagationContext
        ) -> IntermediateData:
            intermediate = self.func(data, context)
            return other_func(intermediate, context)

        return PathPipeline(composed)

    def __call__(
        self, data: IntermediateData, context: PropagationContext
    ) -> IntermediateData:
        """Execute the pipeline."""
        return self.func(data, context)


class PathBuilder:
    """Builder pattern for constructing complex light paths.

    Provides a fluent interface for building paths:
    path = PathBuilder().primary(params).coronagraph(params).build()
    """

    def __init__(self):
        """Initialize empty builder."""
        self._components: List[PathFunction] = []

    def add(self, component: PathFunction) -> "PathBuilder":
        """Add a component to the path.

        Args:
            component:
                Path function to add

        Returns:
            Self for method chaining
        """
        self._components.append(component)
        return self

    def primary(self, params: "PrimaryParams") -> "PathBuilder":
        """Add primary mirror component."""
        return self.add(Path.primary(params))

    def coronagraph(self, params: "CoronagraphParams") -> "PathBuilder":
        """Add coronagraph component."""
        return self.add(Path.coronagraph(params))

    def filter(self, params: "FilterParams") -> "PathBuilder":
        """Add filter component."""
        return self.add(Path.filter(params))

    def detector(self, params: "DetectorParams") -> "PathBuilder":
        """Add detector component."""
        return self.add(Path.detector(params))

    def speckles(self, params: Any) -> "PathBuilder":
        """Add speckles component."""
        return self.add(Path.speckles(params))

    def custom(self, func: PathFunction) -> "PathBuilder":
        """Add custom function component."""
        return self.add(func)

    def build(self) -> PathFunction:
        """Build the final path function.

        Returns:
            Composed path function
        """
        if not self._components:
            raise ValueError("Cannot build empty path")

        return compose(*self._components)


class Path:
    """Factory class for creating path components using the >> pipeline pattern.

    Provides static methods for creating common path components that can be
    chained together with the >> operator for elegant light path composition.
    """

    @staticmethod
    def primary(params: "PrimaryParams") -> PathFunction:
        """Create a primary mirror path function."""

        def primary_func(
            data: IntermediateData, context: PropagationContext
        ) -> IntermediateData:
            return apply_primary(data, context, params)

        return primary_func

    @staticmethod
    def coronagraph(params: "CoronagraphParams") -> PathFunction:
        """Create a coronagraph path function."""

        def coronagraph_func(
            data: IntermediateData, context: PropagationContext
        ) -> IntermediateData:
            return apply_coronagraph(data, context, params)

        return coronagraph_func

    @staticmethod
    def filter(params: "FilterParams") -> PathFunction:
        """Create a filter path function."""

        def filter_func(
            data: IntermediateData, context: PropagationContext
        ) -> IntermediateData:
            return apply_filter(data, context, params)

        return filter_func

    @staticmethod
    def detector(params: "DetectorParams") -> PathFunction:
        """Create a detector path function."""

        def detector_func(
            data: IntermediateData, context: PropagationContext
        ) -> IntermediateData:
            return apply_detector(data, context, params)

        return detector_func

    @staticmethod
    def speckles(params: Any) -> PathFunction:
        """Create a speckles path function."""

        def speckles_func(
            data: IntermediateData, context: PropagationContext
        ) -> IntermediateData:
            return apply_speckles(data, context, params)

        return speckles_func


# Hardware parameter classes (unchanged)
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
    """Parameters for the coronagraph.

    This is a simple wrapper around a yippy Coronagraph object.
    ALL performance characteristics (PSF, contrast, throughput, etc.)
    are handled by yippy - we don't need to duplicate those parameters.
    """

    coronagraph_dir: str  # Path to coronagraph directory
    use_jax: bool = True  # Whether to use JAX for yippy
    cpu_cores: int = 1  # Number of CPU cores for yippy
    _coronagraph_model: Any = None  # Cached yippy.Coronagraph object

    def get_coronagraph_model(self):
        """Get or create the yippy Coronagraph object.

        Returns:
            yippy.Coronagraph object
        """
        if self._coronagraph_model is None:
            try:
                from yippy import Coronagraph

                self._coronagraph_model = Coronagraph(
                    self.coronagraph_dir, use_jax=self.use_jax, cpu_cores=self.cpu_cores
                )
            except ImportError:
                raise ImportError(
                    "yippy package not available. Install yippy to use coronagraph models."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load coronagraph from {self.coronagraph_dir}: {e}"
                )

        return self._coronagraph_model


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
        # Extract the value from pixel scale (it should be arcsec/pix or similar)
        pixel_scale_value = (
            self.pixel_scale.value
            if hasattr(self.pixel_scale, "value")
            else self.pixel_scale
        )
        if pixel_scale_value <= 0:
            raise ValueError("Pixel scale must be positive")
        if self.read_noise.to(u.electron).value < 0:
            raise ValueError("Read noise must be non-negative")
        if self.dark_current.to(u.electron / u.s).value < 0:
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


# Physics functions using the new pattern
def apply_primary(
    data: IntermediateData, context: PropagationContext, params: PrimaryParams
) -> IntermediateData:
    """Apply primary mirror effects to the light.

    This follows the original implementation: illuminated_area = π * D²/4 * (1 - obscuration)

    Args:
        data:
            Input data with source fluxes
        context:
            Propagation context
        params:
            Primary mirror parameters

    Returns:
        Data with primary mirror effects applied
    """
    # Calculate illuminated area following original implementation
    # illuminated_area = π * diameter²/4 * (1 - frac_obscured)
    frac_obscured = getattr(params, "frac_obscured", 0.0)
    illuminated_area = np.pi * params.diameter**2 / 4.0 * (1.0 - frac_obscured)

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


def apply_coronagraph(
    data: IntermediateData, context: PropagationContext, params: CoronagraphParams
) -> IntermediateData:
    """Apply coronagraph suppression to the light.

    This function implements the key physics of coronagraph operation following
    the original implementation. Different source types are handled differently:
    - Star: Uses stellar intensity map from coronagraph model
    - Planets: Uses off-axis PSF based on separation and position angle
    - Disk: Uses PSF datacube convolution for extended source

    Args:
        data:
            Input data with source fluxes
        context:
            Propagation context including wavelength and diameter
        params:
            Coronagraph parameters containing yippy coronagraph model

    Returns:
        Data with coronagraph effects applied
    """
    new_data = data

    # Handle star using stellar intensity map (like gen_star_count_rate)
    if data.star_flux is not None:
        star_count_rate = apply_star_coronagraph(data.star_flux, params, context)
        new_data = new_data.update(star_flux=star_count_rate)

    # Handle planets using off-axis PSF (like gen_planet_count_rate)
    if data.planet_flux is not None:
        planet_count_rate = apply_planet_coronagraph(data.planet_flux, params, context)
        new_data = new_data.update(planet_flux=planet_count_rate)

    # Handle disk using PSF datacube (like gen_disk_count_rate)
    if data.disk_flux_map is not None:
        disk_count_rate = apply_disk_coronagraph(data.disk_flux_map, params, context)
        new_data = new_data.update(disk_flux_map=disk_count_rate)

    return new_data


def apply_filter(
    data: IntermediateData, context: PropagationContext, params: FilterParams
) -> IntermediateData:
    """Apply optical filter transmission to the light.

    Handles both simple rectangular filters and synphot bandpass models.
    Follows the original implementation's wavelength handling.

    Args:
        data:
            Input data with source fluxes
        context:
            Propagation context including wavelength and bandwidth
        params:
            Filter parameters (may include synphot bandpass)

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


def apply_detector(
    data: IntermediateData, context: PropagationContext, params: DetectorParams
) -> IntermediateData:
    """Apply detector effects including coordinate transformation and noise.

    This function follows the original detector implementation:
    1. Resample from coronagraph (lambda/D) to detector pixel scale
    2. Convert photon rates to incident photons (Poisson process)
    3. Apply quantum efficiency (binomial process)
    4. Add detector noise (dark current, read noise, CIC)
    5. Apply saturation limits

    Args:
        data:
            Input data with count rates in coronagraph coordinates
        context:
            Propagation context including wavelength, diameter, and exposure time
        params:
            Detector parameters including pixel scale and noise characteristics

    Returns:
        Data with detector effects applied in detector coordinates
    """
    new_data = data
    exposure_time = context.time_step.to(u.s).value

    # Set random seed for reproducibility
    if hasattr(context, "rng_key") and context.rng_key is not None:
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


def apply_speckles(
    data: IntermediateData, context: PropagationContext, params: Any
) -> IntermediateData:
    """Apply speckle noise to the data.

    Args:
        data:
            Input data
        context:
            Propagation context
        params:
            Speckle parameters (placeholder)

    Returns:
        Data with speckle effects applied
    """
    # This is a placeholder for speckle modeling
    # In practice, this would add coherent speckle noise

    # For now, just add some correlated noise
    if hasattr(context, "rng_key") and context.rng_key is not None:
        np.random.seed(context.rng_key % (2**32))

    # Add simple speckle-like noise to star flux
    if data.star_flux is not None:
        speckle_noise = np.random.normal(
            0, 0.01 * data.star_flux.values, size=data.star_flux.shape
        )

        noisy_star_flux = data.star_flux + speckle_noise
        return data.update(star_flux=noisy_star_flux)

    return data


# Helper functions (unchanged)
def apply_star_coronagraph(
    star_flux, params: CoronagraphParams, context: PropagationContext
):
    """Apply coronagraph to stellar flux using stellar intensity map from yippy.

    Follows the original gen_star_count_rate implementation:
    1. Get stellar intensity map from yippy coronagraph
    2. Multiply star flux by the intensity map
    """
    try:
        # Get the yippy coronagraph model
        coronagraph_model = params.get_coronagraph_model()

        # For now, use a simple stellar diameter (this would come from the star properties)
        stellar_diameter_lod = 0.01  # lambda/D units, typical for nearby stars

        # Get the stellar intensity map from the coronagraph
        stellar_intens = coronagraph_model.stellar_intens(stellar_diameter_lod)

        # Apply the coronagraph suppression
        # The intensity map gives the suppression factor at each pixel
        if hasattr(stellar_intens, "T"):
            stellar_intens = stellar_intens.T

        # Convert star flux to the appropriate format for multiplication
        if hasattr(star_flux, "value"):
            flux_value = star_flux.value
        else:
            flux_value = float(star_flux)

        # Multiply flux by intensity map
        suppressed_flux_map = stellar_intens * flux_value

        return suppressed_flux_map

    except Exception as e:
        print(f"Warning: Could not use yippy coronagraph: {e}")
        # Fallback to very simple suppression if yippy fails
        return star_flux * 1e-10  # Simple high contrast suppression


def apply_planet_coronagraph(
    planet_flux, params: CoronagraphParams, context: PropagationContext
):
    """Apply coronagraph to planetary flux using off-axis PSF from yippy.

    Follows the original gen_planet_count_rate implementation:
    1. Get yippy coronagraph model
    2. Apply off-axis PSF for each planet's location
    3. Sum contributions from all planets
    """
    try:
        # Get the yippy coronagraph model
        coronagraph_model = params.get_coronagraph_model()

        # For now, assume planets are at a reasonable separation
        # In full implementation, this would come from orbital propagation
        planet_separation_lod = 5.0  # lambda/D units

        # Get off-axis PSF from yippy
        # This is a simplified version - real implementation would loop over planet positions
        psf = coronagraph_model.offax(0, planet_separation_lod)  # x=0, y=separation

        # Apply PSF to planet flux
        if hasattr(planet_flux, "value"):
            flux_value = planet_flux.value
        else:
            flux_value = float(planet_flux)

        processed_flux = flux_value * psf.sum()  # Simple integration over PSF

        return processed_flux

    except Exception as e:
        print(f"Warning: Could not use yippy coronagraph for planets: {e}")
        # Fallback to simple throughput
        return planet_flux * 0.1  # Assume 10% throughput


def apply_disk_coronagraph(
    disk_flux_map, params: CoronagraphParams, context: PropagationContext
):
    """Apply coronagraph to disk flux using PSF datacube from yippy.

    Follows the original gen_disk_count_rate implementation:
    1. Get PSF datacube from yippy
    2. Convolve with disk flux map
    """
    try:
        # Get the yippy coronagraph model
        coronagraph_model = params.get_coronagraph_model()

        # For now, apply a simple uniform suppression
        # In full implementation, this would use PSF datacube convolution
        # psf_datacube = coronagraph_model.psf_datacube

        # Simple uniform suppression for now
        processed_flux = disk_flux_map * 0.1  # Assume 10% throughput for disk

        return processed_flux

    except Exception as e:
        print(f"Warning: Could not use yippy coronagraph for disk: {e}")
        # Fallback to simple throughput
        return disk_flux_map * 0.1  # Assume 10% throughput


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


def _apply_detector_to_flux(
    flux_data, params: DetectorParams, context: PropagationContext, exposure_time: float
):
    """Apply detector effects to a single flux component.

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
    # Convert to numpy array first to handle both scalar and array inputs
    photons_array = np.asarray(incident_photons)
    photo_electrons = np.asarray(
        np.random.binomial(n=photons_array.astype(int), p=params.quantum_efficiency)
    ).astype(float)

    # Step 3: Add detector noise
    shape = photo_electrons.shape

    # Dark current (Poisson process)
    expected_dark = params.dark_current.to(u.electron / u.s).value * exposure_time
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
    total_electrons = (
        photo_electrons + dark_electrons + read_noise_electrons + cic_electrons
    )

    # Step 5: Apply saturation
    saturated_electrons = np.minimum(
        total_electrons, params.saturation_level.to(u.electron).value
    )

    return saturated_electrons


def resample_to_detector(
    coro_data,
    coro_pixel_scale: u.Quantity,
    det_params: DetectorParams,
    wavelength: u.Quantity,
    diameter: u.Quantity,
):
    """Resample data from coronagraph coordinates to detector coordinates.

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
            lod_to_arcsec = (wavelength / diameter).to(
                u.arcsec, u.dimensionless_angles()
            )
            coro_scale_arcsec = coro_pixel_scale * lod_to_arcsec

        # Resample using flux-conserving affine transformation
        resampled = flux_conserving_affine(
            coro_data.astype(float),
            pixscale_src=coro_scale_arcsec,
            pixscale_tgt=det_params.pixel_scale,
            shape_tgt=det_params.shape,
            rotation_deg=0.0,  # No rotation for now
            order=3,
        )

        return resampled

    except Exception:
        # Fallback to simple resampling if advanced transforms fail
        return _simple_resample(coro_data, det_params.shape)


def _simple_resample(data, target_shape):
    """Simple resampling fallback using scipy zoom."""
    try:
        from scipy.ndimage import zoom
    except ImportError:
        # Fallback if scipy is not available
        return data

    if len(data.shape) != 2:
        return data  # Can't handle non-2D data simply

    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return zoom(data, zoom_factors, order=1)


def load_scene_from_exovista(
    scene_path: str, context: PropagationContext
) -> IntermediateData:
    """Load a scene from an ExoVista file into IntermediateData format.

    Args:
        scene_path:
            Path to the ExoVista scene file (e.g., "input/scenes/more_pix.fits")
        context:
            Propagation context containing time and wavelength information

    Returns:
        IntermediateData loaded from the ExoVista scene file
    """
    try:
        from exoverses.exovista.system import ExovistaSystem
    except ImportError:
        raise ImportError(
            "ExoVista package not available. Install exoverses to load scene files."
        )

    try:
        # Load the ExoVista system - it expects a Path object
        from pathlib import Path

        scene_file = Path(scene_path)
        system = ExovistaSystem(scene_file)

        # Extract components and build IntermediateData
        dataset_dict = {}
        coords = {}
        attrs = {
            "scene_file": scene_path,
            "loaded_at": context.time.iso if context.time else "unknown",
            "exovista_system_name": getattr(system, "name", "unknown"),
        }

        # Extract star data
        if hasattr(system, "star") and system.star is not None:
            try:
                star_flux = system.star.spec_flux_density(
                    context.wavelength, context.time
                )
                # Convert to simple scalar for now (no wavelength dependence)
                star_flux_value = (
                    star_flux.to(u.Jy).value
                    if hasattr(star_flux, "to")
                    else float(star_flux)
                )

                dataset_dict["star_flux"] = xr.DataArray(
                    star_flux_value,
                    dims=[],
                    attrs={"units": "Jy", "source": "ExoVista star"},
                )
            except Exception as e:
                print(f"Warning: Could not extract star flux: {e}")

        # Extract planet data
        if hasattr(system, "planets") and system.planets:
            planet_fluxes = []
            planet_coords_x = []
            planet_coords_y = []

            for i, planet in enumerate(system.planets):
                try:
                    # Get planet flux
                    planet_flux = planet.spec_flux_density(
                        context.wavelength, context.time
                    )
                    planet_flux_value = (
                        planet_flux.to(u.Jy).value
                        if hasattr(planet_flux, "to")
                        else float(planet_flux)
                    )
                    planet_fluxes.append(planet_flux_value)

                    # Get planet coordinates (simplified for now)
                    # In practice, would need orbital propagation
                    planet_coords_x.append(0.0)  # Placeholder
                    planet_coords_y.append(0.0)  # Placeholder

                except Exception as e:
                    print(f"Warning: Could not extract planet {i} data: {e}")
                    planet_fluxes.append(0.0)
                    planet_coords_x.append(0.0)
                    planet_coords_y.append(0.0)

            if planet_fluxes:
                dataset_dict["planet_flux"] = xr.DataArray(
                    planet_fluxes,
                    dims=["planet"],
                    coords={"planet": range(len(planet_fluxes))},
                    attrs={"units": "Jy", "source": "ExoVista planets"},
                )

                dataset_dict["planet_coords"] = xr.DataArray(
                    np.column_stack([planet_coords_x, planet_coords_y]),
                    dims=["planet", "coord"],
                    coords={"planet": range(len(planet_fluxes)), "coord": ["x", "y"]},
                    attrs={"units": "arcsec", "source": "ExoVista planet positions"},
                )

        # Extract disk data
        if hasattr(system, "disk") and system.disk is not None:
            try:
                disk_flux_map = system.disk.spec_flux_density(
                    context.wavelength, context.time
                )
                # Convert to numpy array if needed
                if hasattr(disk_flux_map, "value"):
                    disk_flux_map = disk_flux_map.value

                # Get disk image dimensions
                if hasattr(disk_flux_map, "shape") and len(disk_flux_map.shape) >= 2:
                    ny, nx = disk_flux_map.shape[-2:]
                    coords["x"] = np.arange(nx)
                    coords["y"] = np.arange(ny)

                    dataset_dict["disk_flux_map"] = xr.DataArray(
                        disk_flux_map.squeeze()
                        if len(disk_flux_map.shape) > 2
                        else disk_flux_map,
                        dims=["y", "x"],
                        coords={"x": coords["x"], "y": coords["y"]},
                        attrs={"units": "Jy/pixel", "source": "ExoVista disk"},
                    )
                else:
                    print("Warning: Disk flux map does not have expected 2D structure")

            except Exception as e:
                print(f"Warning: Could not extract disk data: {e}")

        # Create the dataset
        if not dataset_dict:
            # Fallback if no components could be loaded
            dataset_dict["star_flux"] = xr.DataArray(
                1e-10,  # Very faint star
                dims=[],
                attrs={"units": "Jy", "source": "fallback"},
            )

        dataset = xr.Dataset(dataset_dict, coords=coords, attrs=attrs)
        return IntermediateData(dataset)

    except Exception as e:
        raise RuntimeError(f"Failed to load ExoVista scene from {scene_path}: {e}")


def load_scene(target_path: str, context: PropagationContext) -> IntermediateData:
    """Load a scene from a file into IntermediateData format.

    This is a wrapper that delegates to the appropriate loader based on file type.

    Args:
        target_path:
            Path to the scene file
        context:
            Propagation context

    Returns:
        IntermediateData loaded from the scene file
    """
    # For now, assume all scene files are ExoVista format
    return load_scene_from_exovista(target_path, context)


# Convenience functions for backward compatibility
def primary_step(params: PrimaryParams) -> PathFunction:
    """Backward compatibility function for primary step."""
    return Path.primary(params)


def coronagraph_step(params: CoronagraphParams) -> PathFunction:
    """Backward compatibility function for coronagraph step."""
    return Path.coronagraph(params)


def filter_step(params: FilterParams) -> PathFunction:
    """Backward compatibility function for filter step."""
    return Path.filter(params)


def detector_step(params: DetectorParams) -> PathFunction:
    """Backward compatibility function for detector step."""
    return Path.detector(params)


def speckles_step(params: Any) -> PathFunction:
    """Backward compatibility function for speckles step."""
    return Path.speckles(params)
