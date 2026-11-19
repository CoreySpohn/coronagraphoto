"""The local-zodi sightline is derived from the star's sky position and the time."""

import jax.numpy as jnp
import numpy as np
from conftest import make_path, make_system, sightline, with_zodi
from hwoutils.constants import arcsec2rad
from optixstuff import Exposure
from optixstuff.spectral import Band, Quadrature
from zodi.jax import LeinertZodi

from coronagraphoto import photon_rates

_OBLIQUITY_DEG = 23.4392911


def test_derived_sightline_reaches_pixels():
    """Fails if latitude and longitude are interchanged or sr is treated as arcsec2.

    Independent scene radiance times known transmission and area: 64 eps, with
    the radiance evaluated at ecliptic coordinates computed here from the
    star's RA and Dec and the solar longitude of the exposure time. Three
    times make the geometry live without a large annual image cube.
    """
    zodi = LeinertZodi()
    path = make_path(n=8, step=0.5)
    system = with_zodi(make_system(), zodi)
    ra, dec = np.radians(system.star.ra_deg), np.radians(system.star.dec_deg)
    sin_beta = np.sin(dec) * np.cos(np.radians(_OBLIQUITY_DEG)) - np.cos(dec) * np.sin(
        np.radians(_OBLIQUITY_DEG)
    ) * np.sin(ra)
    beta = float(np.degrees(np.arcsin(sin_beta)))
    values = []
    for day in (0.0, 60.0, 150.0):
        lat, lon = sightline(system, day)
        np.testing.assert_allclose(float(lat), beta, atol=1e-4)
        exposure = Exposure(
            time_d=day,
            roll_deg=0,
            quadrature=Quadrature.build(
                (Band(549.5, 550.5),), system, path, zodi, order=1
            ),
        )
        rate = photon_rates(system, path, exposure).local_zodi[0, 0, 4, 4]
        radiance = zodi.photon_radiance(
            550.0,
            ecliptic_latitude_deg=jnp.asarray(lat),
            solar_longitude_deg=jnp.asarray(lon),
        )
        expected = (
            40
            * 0.4
            * 0.3
            * radiance
            * float(path.detector.pixel_scale_arcsec) ** 2
            * arcsec2rad**2
        )
        np.testing.assert_allclose(rate, expected, rtol=64 * np.finfo(float).eps)
        values.append(float(rate))
    assert np.ptp(values) > 0
