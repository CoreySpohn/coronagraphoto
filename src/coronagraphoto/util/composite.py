"""Utility functions for creating composite exposures and RGB images."""

import jax
import jax.numpy as jnp
import numpy as np
from astropy.visualization import AsinhStretch, MinMaxInterval, make_rgb

from coronagraphoto.core import Exposure


def get_composite_exposure(
    base_exposure: Exposure,
    rgb_centers_nm: jnp.ndarray,
    bandwidth_fraction: float = 0.1,
    n_sub_waves: int = 3,
) -> Exposure:
    """Creates a batched Exposure object for RGB simulation.

    This function expands a single 'base' exposure into a batch of 3 exposures
    (Red, Green, Blue). It handles the creation of sub-wavelengths for
    broadband simulation if n_sub_waves > 1.

    Args:
        base_exposure: An Exposure object containing the common parameters
                       (time, integration time, etc.).
        rgb_centers_nm: An array of 3 central wavelengths (e.g., [650, 550, 450]).
        bandwidth_fraction: The fractional bandwidth (Delta lambda / lambda).
        n_sub_waves: Number of spectral sub-samples per channel for integration.

    Returns:
        Exposure: A batched Exposure object where attributes like
                  `central_wavelength_nm` have shape (3, n_sub_waves).
    """

    # helper to generate sub-wavelengths for a single center
    def make_band(center_wl):
        delta = center_wl * bandwidth_fraction
        # Linspace centered around the wavelength
        return jnp.linspace(center_wl - delta / 2, center_wl + delta / 2, n_sub_waves)

    # Vectorize to get shape (3, n_sub_waves)
    # rgb_centers_nm shape: (3,) -> wavelengths shape: (3, n_sub_waves)
    wavelengths = jax.vmap(make_band)(rgb_centers_nm)

    # Calculate bin widths
    # Shape: (3, n_sub_waves)
    bin_widths = (wavelengths[:, 1] - wavelengths[:, 0])[:, None] * jnp.ones_like(
        wavelengths
    )

    # Broadcast the time to match the batch size (3)
    # If base_exposure.start_time_jd is a scalar, we repeat it.
    times = jnp.repeat(base_exposure.start_time_jd, 3)

    return Exposure(
        start_time_jd=times,
        exposure_time_s=base_exposure.exposure_time_s,
        central_wavelength_nm=wavelengths,
        bin_width_nm=bin_widths,
        position_angle_deg=base_exposure.position_angle_deg,
    )


def render_rgb(image_batch: np.ndarray, interval=None, stretch=None) -> np.ndarray:
    """Converts a raw simulation batch (3, H, W) into a visualizable RGB image.

    Args:
        image_batch: Numpy array of shape (3, height, width).
                     Index 0=R, 1=G, 2=B.
        interval: Astropy Interval object (defaults to MinMaxInterval).
        stretch: Astropy Stretch object (defaults to AsinhStretch).

    Returns:
        np.ndarray: An (H, W, 3) array of floats [0, 1] ready for plt.imshow.
    """
    if interval is None:
        interval = MinMaxInterval()
    if stretch is None:
        stretch = AsinhStretch(a=0.1)

    # Ensure input is on CPU/Numpy for Astropy
    image_batch = np.array(image_batch)

    return make_rgb(
        image_batch[0],  # Red
        image_batch[1],  # Green
        image_batch[2],  # Blue
        interval=interval,
        stretch=stretch,
    )
