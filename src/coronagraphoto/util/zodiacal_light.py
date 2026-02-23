"""Zodiacal light utilities for coronagraphoto.

This module re-exports zodiacal light functions from ``orbix.observatory.zodiacal``.
All functions are JAX-compatible for JIT compilation.

Two approaches are supported:

1. **AYO Default**: Fixed 22 mag/arcsec^2 at V-band (assuming 135 deg solar longitude)
   with wavelength-dependent color correction. This is the simplest approach
   and matches the convention used in the ETC Calibration Task Group study.

2. **Leinert Tables**: Full position-dependent zodiacal light using the
   Leinert et al. (1998) model with interpolation in ecliptic coordinates.
   This provides more accurate zodi estimates but requires target coordinates.

Note: coronagraphoto treats zodiacal light surface brightness as an INPUT rather
than calculating it internally. This keeps telescope pointing considerations
outside of coronagraphoto's scope, while still providing utility functions
for users who want to calculate zodi values themselves.

References:
    Leinert, C., et al. (1998). The 1997 reference of diffuse night sky brightness.
    Astronomy and Astrophysics Supplement Series, 127(1), 1-99.

    Stark, C., et al. (2019). AYO exposure time methods.

    ETC Calibration Paper (2025). arxiv:2502.18556

Note:
    The canonical implementation now lives in ``orbix.observatory.zodiacal``.
    This module re-exports those functions for backward compatibility.
"""

# Re-export everything from orbix
from orbix.observatory.zodiacal import (
    AB_ZERO_POINT_JY,
    AYO_DEFAULT_ZODI_MAG_V,
    JOHNSON_V_ZERO_POINT_JY,
    LEINERT_B_LAMBDA,
    LEINERT_BETA_DEG,
    LEINERT_SOLAR_LON_DEG,
    LEINERT_TABLE17,
    LEINERT_WAVELENGTH_UM,
    V_BAND_WAVELENGTH_NM,
    ayo_default_zodi_flux_jy,
    ayo_default_zodi_mag,
    create_zodi_spectrum_jax,
    flux_to_mag_jy,
    leinert_zodi_factor,
    leinert_zodi_mag,
    leinert_zodi_spectral_radiance,
    mag_to_flux_jy,
    zodi_color_correction,
)

import jax
import jax.numpy as jnp


def create_zodi_spectrum(
    wavelengths_nm: jnp.ndarray,
    ecliptic_lat_deg: float = 0.0,
    solar_lon_deg: float = 135.0,
    use_ayo_default: bool = True,
) -> jnp.ndarray:
    """Create a zodiacal light spectrum in Jy/arcsec^2.

    Args:
        wavelengths_nm: Array of wavelengths in nanometers.
        ecliptic_lat_deg: Ecliptic latitude in degrees.
        solar_lon_deg: Solar longitude in degrees.
        use_ayo_default: If True, use AYO default (22 mag/arcsec^2 at V).
                        If False, use full Leinert table calculation.

    Returns:
        Array of surface brightness values in Jy/arcsec^2.
    """

    def _compute_flux_ayo(wl):
        return ayo_default_zodi_flux_jy(wl)

    def _compute_flux_leinert(wl):
        mag = leinert_zodi_mag(wl, ecliptic_lat_deg, solar_lon_deg)
        return mag_to_flux_jy(mag)

    if use_ayo_default:
        flux = jax.vmap(_compute_flux_ayo)(wavelengths_nm)
    else:
        flux = jax.vmap(_compute_flux_leinert)(wavelengths_nm)

    return flux
