"""Observatory pointing reaches the local-zodi photon reader."""

import jax.numpy as jnp
import numpy as np
from conftest import make_path, make_system
from hwoutils.constants import arcsec2rad
from hwoutils.grids import SkyOrientation
from hwoutils.spectral import Band
from orbix.observatory import ObservatoryL2Halo
from zodi.jax import LeinertZodi

from coronagraphoto import image_rates


def test_observatory_zodi_radiance_reaches_pixels():
    """Fails if latitude/longitude are interchanged or sr is treated as arcsec2.

    Independent scene radiance times known transmission/area: 64 eps.
    Three epochs make the geometry live without a large annual image cube.
    """
    orbit = ObservatoryL2Halo.from_default(equinox_mjd=60575.25)
    zodi = LeinertZodi()
    path = make_path(n=8, step=0.5)
    system = make_system()
    values = []
    for day in (0.0, 60.0, 150.0):
        mjd = 60575.25 + day
        lat = orbit.ecliptic_latitude_deg(mjd, jnp.array(1.0), jnp.array(0.4))
        lon = orbit.helio_ecliptic_longitude_deg(mjd, jnp.array(1.0), jnp.array(0.4))
        rate = image_rates(
            system,
            path,
            (Band(549.5, 550.5),),
            time_d=day,
            pointing=SkyOrientation(0),
            quadrature_order=1,
            zodi=zodi,
            ecliptic_latitude_deg=lat,
            solar_longitude_deg=lon,
        ).local_zodi[0, 0, 4, 4]
        radiance = zodi.photon_radiance(
            550.0, ecliptic_latitude_deg=lat, solar_longitude_deg=lon
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
