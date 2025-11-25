"""Detector models."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array

from coronagraphoto import conversions as conv
from coronagraphoto.transforms.image_transforms import resample_flux


# Pure functions for noise simulation
def simulate_dark_current(
    dark_current_rate: float,
    exposure_time: float,
    shape: tuple[int, int],
    prng_key: PRNGKey,
) -> Array:
    """Simulate dark current.

    Args:
        dark_current_rate: Dark current rate in electrons/s/pixel.
        exposure_time: Exposure time in seconds.
        shape: Detector shape (nx, ny).
        prng_key: Random number key.

    Returns:
        Number of dark current electrons.
    """
    return jax.random.poisson(prng_key, dark_current_rate * exposure_time, shape=shape)


def simulate_cic(
    cic_rate: float,
    num_frames: int,
    shape: tuple[int, int],
    prng_key: PRNGKey,
) -> Array:
    """Simulate clock-induced charge (CIC).

    I honestly don't know if this is the right way to do this.

    Args:
        cic_rate: CIC rate in electrons/pixel/frame.
        num_frames: Number of frames (reads).
        shape: Detector shape (nx, ny).
        prng_key: Random number key.

    Returns:
        Number of CIC electrons.
    """
    # CIC is per frame, so total CIC is Poisson(cic_rate * num_frames)
    return jax.random.poisson(prng_key, cic_rate * num_frames, shape=shape)


def simulate_read_noise(
    read_noise: float,
    num_frames: int,
    shape: tuple[int, int],
    prng_key: PRNGKey,
) -> Array:
    """Simulate read noise.

    Args:
        read_noise: Read noise in electrons/pixel/read.
        num_frames: Number of frames (reads).
        shape: Detector shape (nx, ny).
        prng_key: Random number key.

    Returns:
        Read noise electrons.
    """
    # Read noise is Gaussian. Variance adds.
    # Total read noise sigma = sqrt(num_frames) * read_noise
    sigma = read_noise * jnp.sqrt(num_frames)
    return sigma * jax.random.normal(prng_key, shape=shape)


class AbstractDetector(eqx.Module):
    """Abstract base class for all detectors."""

    def resample_to_detector(
        self,
        inc_flux: Array,
        inc_pixel_scale: float,
        wavelength: float,
        primary_diameter_m: float,
    ):
        """Resample the incident flux to the detector plane.

        Args:
            inc_flux: The incident flux in photons/s after passing through the coronagraph and color filter.
            inc_pixel_scale: The pixel scale of the incident image from the coronagraph in (lambda/D)/pixel.
            wavelength: The wavelength in nm (placeholder for consistency).
            primary_diameter_m: The diameter of the primary mirror in meters.
        """
        # Convert the pixel scale of the incident image to arcsec/pixel
        inc_pixel_scale_arcsec = conv.lambda_d_to_arcsec(
            inc_pixel_scale, wavelength, primary_diameter_m
        )

        # Rotation should be applied prior to the source entering the coronagraph,
        # not when it is mapped to the detector plane.
        rotation_deg = 0.0

        # Map the incident flux to the detector plane
        return resample_flux(
            inc_flux, inc_pixel_scale_arcsec, self.pixel_scale, self.shape, rotation_deg
        )

    def readout_source_electrons(
        self, inc_photon_rate: Array, exposure_time: float, prng_key: PRNGKey
    ):
        """Map incident photons onto the detector plane and then read electrons.

        Args:
            inc_photon_rate: The incident photon rate in photons/s after passing through the coronagraph and color filter.
            exposure_time: The exposure time in seconds.
            prng_key: The PRNG key for the random number generator.

        Returns:
            The readout electrons in electrons.
        """
        # Simulate photon conversion
        key_phot, key_qe = jax.random.split(prng_key, 2)

        # Convert photon rate to photons
        inc_photons = jax.random.poisson(key_phot, inc_photon_rate * exposure_time)

        # Convert photons to photo-electrons with QE
        photo_electrons = jax.random.binomial(
            key_qe, inc_photons, self.quantum_efficiency
        )
        return photo_electrons

    def readout_noise_electrons(self, exposure_time: float, prng_key: PRNGKey):
        """Map incident photons onto the detector plane and then read electrons."""
        raise NotImplementedError


@final
class SimpleDetector(AbstractDetector):
    """Simple detector model with no wavelength dependence."""

    pixel_scale: float  # arcsec/pixel
    shape: tuple[int, int]  # (nx, ny)
    # fraction of incident photons converted to photo-electrons
    quantum_efficiency: float

    # electrons/s/pixel
    dark_current_rate: float

    def __init__(
        self,
        pixel_scale: float,
        shape: tuple[int, int],
        quantum_efficiency: float = 1.0,
        dark_current_rate: float = 0.0,
    ):
        """Initialize the simple detector."""
        self.pixel_scale = pixel_scale
        self.shape = shape
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate = dark_current_rate

    def readout_noise_electrons(self, exposure_time: float, prng_key: PRNGKey):
        """Read out the dark current electrons."""
        # Add dark current
        dark_current = simulate_dark_current(
            self.dark_current_rate, exposure_time, self.shape, prng_key
        )

        return dark_current


@final
class Detector(AbstractDetector):
    """Detector model with CIC and read noise."""

    pixel_scale: float  # arcsec/pixel
    shape: tuple[int, int]  # (nx, ny)
    # fraction of incident photons converted to photo-electrons
    quantum_efficiency: float

    # electrons/s/pixel
    dark_current_rate: float
    # electrons/readout/pixel
    read_noise: float
    # electrons/frame/pixel
    cic_rate: float
    # time per frame in seconds
    frame_time: float

    def __init__(
        self,
        pixel_scale: float,
        shape: tuple[int, int],
        quantum_efficiency: float = 1.0,
        dark_current_rate: float = 0.0,
        read_noise: float = 0.0,
        cic_rate: float = 0.0,
        frame_time: float = 1.0,
    ):
        """Initialize the detector."""
        self.pixel_scale = pixel_scale
        self.shape = shape
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate = dark_current_rate
        self.read_noise = read_noise
        self.cic_rate = cic_rate
        self.frame_time = frame_time

    def readout_noise_electrons(self, exposure_time: float, prng_key: PRNGKey):
        """Map incident photons onto the detector plane and then read electrons.

        Args:
            exposure_time: The exposure time in seconds.
            prng_key: The PRNG key for the random number generator.

        Returns:
            The readout electrons in electrons.
        """
        # Split keys for different noise sources
        key_dark, key_cic, key_read = jax.random.split(prng_key, 3)

        # Add dark current
        dark_current = simulate_dark_current(
            self.dark_current_rate, exposure_time, self.shape, key_dark
        )

        # Calculate number of frames
        num_frames = jnp.ceil(exposure_time / self.frame_time)

        # Add CIC
        cic = simulate_cic(self.cic_rate, num_frames, self.shape, key_cic)

        # Add Read Noise
        read_noise = simulate_read_noise(
            self.read_noise, num_frames, self.shape, key_read
        )

        return dark_current + cic + read_noise
