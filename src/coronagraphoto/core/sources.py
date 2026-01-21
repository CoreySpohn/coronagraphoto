"""Refactored astrophysical source objects with no control flow."""

import abc
from typing import final

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from orbix.equations.orbit import mean_anomaly_tp
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.system.planets import Planets as OrbixPlanet

from coronagraphoto import conversions as conv

# Trig solver for orbital propagation (scalar function, will be vectorized in Planets)
TRIG_SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)


# Abstract base class defining the interface for all sources
class AbstractSource(eqx.Module):
    """Abstract base class defining the interface for all astrophysical sources.

    All sources accept scalar wavelength and time inputs. Use jax.vmap
    for vectorized evaluation over multiple wavelengths/times.

    Following Equinox's abstract/final pattern:
    - This class is abstract (cannot be instantiated)
    - Concrete subclasses are final (should not be subclassed)
    - All methods are implemented precisely once (no overriding)
    """

    @abc.abstractmethod
    def spec_flux_density(self, wavelength: float, time: float) -> float | jnp.ndarray:
        """Return spectral flux density in ph/s/m^2/nm.

        Args:
            wavelength:
              Scalar wavelength in nm
            time:
              Scalar time in Julian days

        Returns:
            Scalar flux (for point sources) or 2D array (for extended sources)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def spatial_extent(self) -> tuple[float, float] | None:
        """Return spatial extent in arcsec (width, height) or None for point sources.

        Returns:
            Tuple of (width, height) in arcseconds, or None for point sources.
        """
        raise NotImplementedError


@final
class StarSource(AbstractSource):
    """On-axis stellar source."""

    diameter_arcsec: float
    mass_kg: float
    dist_pc: float
    midplane_pa_deg: float
    midplane_i_deg: float
    _wavelengths_nm: jnp.ndarray
    _times_jd: jnp.ndarray
    _flux_density_phot: jnp.ndarray  # Already in ph/s/m^2/nm
    _flux_interp: callable

    def __init__(
        self,
        diameter_arcsec: float,
        mass_kg: float,
        dist_pc: float,
        midplane_pa_deg: float,
        midplane_i_deg: float,
        wavelengths_nm: jnp.ndarray,
        times_jd: jnp.ndarray,
        flux_density_jy: jnp.ndarray,
    ):
        """Initialize the StarSource.

        Args:
            diameter_arcsec: The angular diameter of the star in arcseconds.
            mass_kg: The mass of the star in kg.
            dist_pc: The distance to the star in parsecs.
            midplane_pa_deg: The position angle of the system midplane in degrees.
            midplane_i_deg: The inclination of the system midplane in degrees.
            wavelengths_nm: The wavelengths at which the flux density is provided, in nm.
            times_jd: The times at which the flux density is provided, in Julian days.
            flux_density_jy: The flux density of the star in Janskys.
        """
        self.diameter_arcsec = diameter_arcsec
        self.mass_kg = mass_kg
        self.dist_pc = dist_pc
        self.midplane_pa_deg = midplane_pa_deg
        self.midplane_i_deg = midplane_i_deg
        self._wavelengths_nm = wavelengths_nm
        self._times_jd = times_jd
        # Convert to photons on initialization
        self._flux_density_phot = jax.vmap(
            conv.jy_to_photons_per_nm_per_m2, in_axes=(1, None), out_axes=1
        )(flux_density_jy, wavelengths_nm)
        # Create interpolator2D with cubic method
        self._flux_interp = interpax.Interpolator2D(
            wavelengths_nm, times_jd, self._flux_density_phot, method="cubic"
        )

    def spec_flux_density(self, wavelength: float, time: float) -> float:
        """Get spectral flux density at given wavelength and time.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days

        Returns:
            Scalar flux density in ph/s/m²/nm
        """
        flux = self._flux_interp(wavelength, time)
        return flux

    def spatial_extent(self) -> tuple[float, float] | None:
        """Stars have finite angular size."""
        return (self.diameter_arcsec, self.diameter_arcsec)


@final
class PlanetSources(AbstractSource):
    """Represents a collection of planets as point sources.

    Attributes:
        star: The host star of the planets.
        contrast_interp: A 3D interpolator for the planets' contrast as a
            function of wavelength (nm) and mean anomaly (deg). The last
            dimension of the interpolation array corresponds to the planet index.
            The first dimension is the wavelength, the second is the mean anomaly,
            and the third is the planet index.
        orbix_planets: An Orbix `Planets` object representing the orbital
            properties of all planets.
    """

    star: StarSource
    contrast_interp: interpax.Interpolator3D
    orbix_planets: OrbixPlanet
    n_planets: int

    def __init__(
        self,
        star: StarSource,
        contrast_interp: interpax.Interpolator3D,
        orbix_planets: OrbixPlanet,
    ):
        """Initialize the PlanetSources."""
        self.star = star
        self.contrast_interp = contrast_interp
        self.orbix_planets = orbix_planets
        self.n_planets = orbix_planets.a.shape[0]

    def mean_anomaly(self, time_jd: float):
        """Calculate the mean anomalies of the planets at a given time.

        Args:
            time_jd: The time in Julian days.

        Returns:
            The mean anomalies of the planets in degrees.
        """
        return jnp.rad2deg(
            mean_anomaly_tp(time_jd, self.orbix_planets.n, self.orbix_planets.tp)
            % (2 * jnp.pi)
        )

    def contrast(self, wavelength_nm: float, time_jd: float):
        """Calculate the contrasts of the planets at a given wavelength and time.

        Args:
            wavelength_nm: The wavelength in nm.
            time_jd: The time in Julian days.

        Returns:
            The contrasts of the planets.
        """
        mean_anomalies_deg = self.mean_anomaly(time_jd)
        planet_indices = jnp.arange(self.n_planets)

        # vmap so that all planets are evaluated at the same wavelength
        # without broadcasting
        interp = jax.vmap(self.contrast_interp, in_axes=(None, 0, 0))

        return interp(wavelength_nm, mean_anomalies_deg, planet_indices)

    def spec_flux_density(self, wavelength_nm: float, time_jd: float) -> jnp.ndarray:
        """Get spectral flux density by multiplying contrast with star flux.

        Args:
            wavelength_nm: Scalar wavelength in nm
            time_jd: Scalar time in Julian days

        Returns:
            Array of shape (n_planets,) with flux densities in ph/s/m²/nm
        """
        contrast = self.contrast(wavelength_nm, time_jd)
        star_flux = self.star.spec_flux_density(wavelength_nm, time_jd)
        return contrast * star_flux

    def position(self, time_jd: float) -> jnp.ndarray:
        """Calculate on-sky position (dRA, dDec) in arcseconds.

        Args:
            time_jd: Scalar time in Julian days

        Returns:
            Array of shape (2, n_planets) with (dRA, dDec) in arcseconds
        """
        # Propagate orbit
        dra, ddec = self.orbix_planets.prop_ra_dec(TRIG_SOLVER, jnp.atleast_1d(time_jd))

        # Stack to get (2, 1, n_planets) and reshape to (2, n_planets)
        # We use reshape instead of squeeze to preserve the planet dimension
        # even if there is only one planet.
        return jnp.stack([dra, ddec]).reshape(2, self.n_planets)

    def alpha_dMag(self, time_jd: float) -> jnp.ndarray:
        """Calculate the apparent angular separation and dMag of the planets at a given time.

        Args:
            time_jd: The time in Julian days.

        Returns:
            The apparent angular separation and dMag of the planets.
        """
        alpha, dMag = self.orbix_planets.alpha_dMag(
            TRIG_SOLVER, jnp.atleast_1d(time_jd)
        )
        return alpha, dMag

    def spatial_extent(self) -> tuple[float, float] | None:
        """Planets are point sources."""
        raise NotImplementedError


@final
class DiskSource(AbstractSource):
    """Extended disk source (debris disk, exozodiacal light, etc.)."""

    pixel_scale_arcsec: float
    star: StarSource  # Reference to the host star
    _wavelengths_nm: jnp.ndarray  # Shape: (n_wavelengths,)
    _contrast_cube: jnp.ndarray  # Shape: (n_wavelengths, ny, nx)
    _contrast_interp: interpax.CubicSpline

    def __init__(
        self,
        pixel_scale_arcsec: float,
        star: StarSource,
        wavelengths_nm: jnp.ndarray,
        contrast_cube: jnp.ndarray,
    ):
        """Initialize the DiskSource.

        Args:
            pixel_scale_arcsec: The pixel scale of the contrast cube in arcsec/pixel.
            star: The host star.
            wavelengths_nm: The wavelengths at which the contrast is provided, in nm.
            contrast_cube: The contrast of the disk relative to the star.
        """
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.star = star
        self._wavelengths_nm = wavelengths_nm
        self._contrast_cube = contrast_cube
        self._contrast_interp = interpax.CubicSpline(
            wavelengths_nm, contrast_cube, axis=0
        )

    def spec_flux_density(self, wavelength: float, time: float) -> jnp.ndarray:
        """Get spectral flux density map at scalar wavelength and time.

        Implements logarithmic interpolation in wavelength space (matching ExoVista
        pattern) using cubic interpolation along the wavelength axis.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days

        Returns:
            2D array of flux density in ph/s/m2/nm with shape (ny, nx)
        """
        # Interpolate contrast cube along wavelength axis using cubic interpolation
        contrast = self._contrast_interp(wavelength)

        # Get star flux and multiply
        star_flux = self.star.spec_flux_density(wavelength, time)
        disk_flux = contrast * star_flux

        return disk_flux

    def spatial_extent(self) -> tuple[float, float] | None:
        """Return spatial extent of disk in arcseconds."""
        ny, nx = self._contrast_cube.shape[-2:]
        width = nx * self.pixel_scale_arcsec
        height = ny * self.pixel_scale_arcsec
        return (width, height)


@final
class ZodiSource(AbstractSource):
    """Uniform zodiacal light source."""

    _wavelengths_nm: jnp.ndarray
    _flux_density_phot: jnp.ndarray  # ph/s/m^2/nm/arcsec^2
    _flux_interp: interpax.Interpolator1D

    def __init__(
        self,
        wavelengths_nm: jnp.ndarray,
        flux_density_jy_arcsec2: jnp.ndarray,
    ):
        """Initialize the ZodiSource.

        Args:
            wavelengths_nm: The wavelengths at which the flux density is provided, in nm.
            flux_density_jy_arcsec2: The flux density of the source in Janskys/arcsec^2.
        """
        self._wavelengths_nm = wavelengths_nm
        # Convert to photons on initialization
        self._flux_density_phot = conv.jy_to_photons_per_nm_per_m2(
            flux_density_jy_arcsec2, wavelengths_nm
        )
        # Create interpolator1D with linear method
        self._flux_interp = interpax.Interpolator1D(
            wavelengths_nm, self._flux_density_phot, method="linear"
        )

    def spec_flux_density(self, wavelength: float, time: float) -> float:
        """Get spectral flux density at given wavelength.

        Args:
            wavelength: Scalar wavelength in nm
            time: Scalar time in Julian days (ignored for Zodi)

        Returns:
            Scalar flux density in ph/s/m²/nm/arcsec²
        """
        flux = self._flux_interp(wavelength)
        return flux

    def spatial_extent(self) -> tuple[float, float] | None:
        """Zodiacal light fills the field of view."""
        return None
