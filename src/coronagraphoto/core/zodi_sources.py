"""Zodiacal light source objects with position-dependent brightness support."""

import abc
from typing import final

import equinox as eqx
import interpax
import jax.numpy as jnp
from hwoutils.constants import Jy, h
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from orbix.observatory.zodiacal import (
    create_zodi_spectrum_jax,
    leinert_zodi_mag,
    mag_to_flux_jy,
    zodi_color_correction,
)


class AbstractZodiSource(eqx.Module):
    """Abstract base class for zodiacal light sources.

    Zodiacal light sources have an extended interface that includes
    position parameters for computing position-dependent brightness.

    All ZodiSource classes accept optional ecliptic coordinates in
    spec_flux_density to support dynamic position-dependent queries.
    """

    @abc.abstractmethod
    def spec_flux_density(
        self,
        wavelength: float,
        time: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Return spectral flux density in ph/s/m^2/nm/arcsec^2.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days
            ecliptic_lat_deg: Ecliptic latitude in degrees
            solar_lon_deg: Solar longitude in degrees

        Returns:
            Scalar flux density in ph/s/m^2/nm/arcsec^2
        """
        raise NotImplementedError


@final
class ZodiSourceAYO(AbstractZodiSource):
    """Zodiacal light using AYO-compatible default settings.

    Uses a fixed surface brightness at V-band with solar-like color correction
    based on the Leinert et al. (1998) wavelength dependence. This matches the
    ETC Calibration Task Group methodology (135 deg solar longitude assumption).

    This is the recommended choice for benchmarking against AYO/EXOSIMS/pyEDITH.

    Example:
        >>> wavelengths = jnp.linspace(400, 1000, 50)
        >>> zodi = ZodiSourceAYO(wavelengths, surface_brightness_mag=22.0)
    """

    _wavelengths_nm: jnp.ndarray
    _flux_density_phot: jnp.ndarray  # ph/s/m^2/nm/arcsec^2
    _flux_interp: interpax.Interpolator1D
    _reference_wavelength_nm: float
    _reference_mag_arcsec2: float

    def __init__(
        self,
        wavelengths_nm: jnp.ndarray,
        surface_brightness_mag: float = 22.0,
        reference_wavelength_nm: float = 550.0,
    ):
        """Initialize ZodiSourceAYO.

        Args:
            wavelengths_nm: Array of wavelengths in nm.
            surface_brightness_mag: Surface brightness at V-band in mag/arcsec^2.
                                   Default is 22.0 (AYO standard for coronagraphs).
            reference_wavelength_nm: Reference wavelength in nm (default 550 = V-band).
        """
        self._wavelengths_nm = wavelengths_nm
        self._reference_wavelength_nm = reference_wavelength_nm
        self._reference_mag_arcsec2 = surface_brightness_mag

        # Calculate reference flux from magnitude
        reference_flux_jy = mag_to_flux_jy(surface_brightness_mag)

        # Create wavelength-dependent spectrum with color correction
        flux_spectrum_jy = create_zodi_spectrum_jax(
            wavelengths_nm,
            reference_flux_jy=reference_flux_jy,
            reference_wavelength_nm=reference_wavelength_nm,
        )

        # Convert to photons
        self._flux_density_phot = jy_to_photons_per_nm_per_m2(
            flux_spectrum_jy, wavelengths_nm
        )
        self._flux_interp = interpax.Interpolator1D(
            wavelengths_nm, self._flux_density_phot, method="linear"
        )

    @property
    def reference_wavelength_nm(self) -> float:
        """Reference wavelength for the zodi model in nanometers."""
        return self._reference_wavelength_nm

    @property
    def reference_mag_arcsec2(self) -> float:
        """Surface brightness at reference wavelength in mag/arcsec^2."""
        return self._reference_mag_arcsec2

    def spec_flux_density(
        self,
        wavelength: float,
        time: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Get spectral flux density at given wavelength.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days (ignored for Zodi)
            ecliptic_lat_deg: Ignored (AYO uses fixed-angle assumption)
            solar_lon_deg: Ignored (AYO uses fixed-angle assumption)

        Returns:
            Scalar flux density in ph/s/m^2/nm/arcsec^2
        """
        return self._flux_interp(wavelength)


@final
class ZodiSourceLeinert(AbstractZodiSource):
    """Zodiacal light using full Leinert table position-dependent model.

    Uses the Leinert et al. (1998) tables for both position and wavelength
    dependence. Computes flux dynamically based on ecliptic position.

    This class can be reused across multiple observations at different positions.
    The ecliptic coordinates are passed to spec_flux_density at query time.

    Example:
        >>> zodi = ZodiSourceLeinert(reference_mag=22.0)
        >>> # Query flux at different positions
        >>> flux1 = zodi.spec_flux_density(550.0, 0.0, ecliptic_lat_deg=30.0)
        >>> flux2 = zodi.spec_flux_density(550.0, 0.0, ecliptic_lat_deg=45.0)
    """

    _reference_wavelength_nm: float
    _reference_mag_arcsec2: float

    def __init__(
        self,
        reference_mag_arcsec2: float = 22.0,
        reference_wavelength_nm: float = 550.0,
    ):
        """Initialize ZodiSourceLeinert.

        Args:
            reference_mag_arcsec2: Reference surface brightness in mag/arcsec^2.
                                   This is the magnitude at the reference position
                                   (ecliptic pole, solar longitude 90 deg).
            reference_wavelength_nm: Reference wavelength in nm (default V-band).
        """
        self._reference_mag_arcsec2 = reference_mag_arcsec2
        self._reference_wavelength_nm = reference_wavelength_nm

    @property
    def reference_wavelength_nm(self) -> float:
        """Reference wavelength for the zodi model in nanometers."""
        return self._reference_wavelength_nm

    @property
    def reference_mag_arcsec2(self) -> float:
        """Surface brightness at reference wavelength in mag/arcsec^2."""
        return self._reference_mag_arcsec2

    def spec_flux_density(
        self,
        wavelength: float,
        time: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Get spectral flux density at given wavelength and position.

        Computes the zodiacal light flux dynamically using Leinert et al. (1998)
        tables for position and wavelength dependence.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days (ignored for Zodi)
            ecliptic_lat_deg: Ecliptic latitude in degrees
            solar_lon_deg: Solar longitude in degrees

        Returns:
            Scalar flux density in ph/s/m^2/nm/arcsec^2
        """
        # Get position-dependent magnitude at V-band
        position_mag = leinert_zodi_mag(
            self._reference_wavelength_nm, ecliptic_lat_deg, solar_lon_deg
        )

        # Convert magnitude to Jy
        flux_jy_ref = mag_to_flux_jy(position_mag)

        # Apply wavelength-dependent color correction (photon units)
        color_correction = zodi_color_correction(
            wavelength, self._reference_wavelength_nm, photon_units=True
        )

        # Convert Jy to photon flux at this wavelength
        # Using the same conversion as in create_zodi_spectrum_jax
        flux_jy = (
            flux_jy_ref
            * color_correction
            * (wavelength / self._reference_wavelength_nm) ** 2
        )

        # Convert Jy to ph/s/m^2/nm using conversions module
        flux_phot = flux_jy * Jy / (wavelength * h)

        return flux_phot


@final
class ZodiSourcePhotonFlux(AbstractZodiSource):
    """Zodiacal light from pre-computed photon flux values.

    Use this when you have zodiacal light flux values that were already
    calculated by another tool (e.g., EXOSIMS, pyEDITH). This ensures
    coronagraphoto uses exactly the same values without recalculation.

    Example:
        # After running EXOSIMS, extract the zodi flux it calculated
        >>> exosims_flux = exosims_result.zodi_flux  # ph/s/m^2/nm/arcsec^2
        >>> zodi = ZodiSourcePhotonFlux(wavelengths, exosims_flux)
    """

    _wavelengths_nm: jnp.ndarray
    _flux_density_phot: jnp.ndarray  # ph/s/m^2/nm/arcsec^2
    _flux_interp: interpax.Interpolator1D
    _reference_wavelength_nm: float
    _reference_mag_arcsec2: float

    def __init__(
        self,
        wavelengths_nm: jnp.ndarray,
        flux_phot_per_arcsec2: jnp.ndarray,
        reference_mag_arcsec2: float = 22.0,
    ):
        """Initialize ZodiSourcePhotonFlux.

        Args:
            wavelengths_nm: Array of wavelengths in nm.
            flux_phot_per_arcsec2: Array of photon flux values in ph/s/m^2/nm/arcsec^2.
                                   Must have same length as wavelengths_nm.
            reference_mag_arcsec2: Reference magnitude for metadata (default 22.0).
        """
        self._wavelengths_nm = jnp.asarray(wavelengths_nm)
        self._flux_density_phot = jnp.asarray(flux_phot_per_arcsec2)
        self._reference_wavelength_nm = 550.0
        self._reference_mag_arcsec2 = reference_mag_arcsec2

        self._flux_interp = interpax.Interpolator1D(
            self._wavelengths_nm, self._flux_density_phot, method="linear"
        )

    @property
    def reference_wavelength_nm(self) -> float:
        """Reference wavelength for the zodi model in nanometers."""
        return self._reference_wavelength_nm

    @property
    def reference_mag_arcsec2(self) -> float:
        """Surface brightness at reference wavelength in mag/arcsec^2."""
        return self._reference_mag_arcsec2

    def spec_flux_density(
        self,
        wavelength: float,
        time: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Get spectral flux density at given wavelength.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days (ignored for Zodi)
            ecliptic_lat_deg: Ignored (uses pre-computed values)
            solar_lon_deg: Ignored (uses pre-computed values)

        Returns:
            Scalar flux density in ph/s/m^2/nm/arcsec^2
        """
        return self._flux_interp(wavelength)
