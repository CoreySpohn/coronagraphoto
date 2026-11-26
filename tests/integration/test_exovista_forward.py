"""Synthetic FITS scene loading through photon imaging."""

import numpy as np
from conftest import exposure_for, make_path, write_scene
from hwoutils.constants import arcsec2rad
from hwoutils.numerics import roundoff
from optixstuff.grids import SkyOrientation
from optixstuff.spectral import Band, Quadrature
from skyscapes import from_exovista

from coronagraphoto import photon_rates


def test_exovista_native_disk_and_position_angle(tmp_path):
    """Fails on a zodi sightline not derived from the star, or a cube turned twice.

    Stored contrast rounds to float32 upstream; its radiance identity has
    a 32-operation bound for spline and radiometric arithmetic after that
    rounding. The FITS epoch position is independently known in float64 from
    the 1 AU westward state and 10 pc distance. The 64-operation positional
    bound covers Cartesian-to-orbital conversion, propagation and projection.
    """
    scene = from_exovista(write_scene(tmp_path / "scene.fits"))
    system = scene
    offsets = scene.positions(np.array([0.0]))[:, 0, 0]
    np.testing.assert_allclose(
        offsets,
        [-0.1, 0.0],
        rtol=0,
        atol=roundoff(0.1, n_ops=64, dtype=offsets.dtype),
    )
    assert system.midplane_position_angle_deg == 120.0
    grid = system.disk.native_grid
    assert grid.shape == (12, 12) and grid.step == 0.01
    assert grid.orientation == SkyOrientation(0)
    phi = system.star.photon_flux_density(550.0, 0.0, dist_pc=system.dist_pc)
    radiance = system.photon_radiance(550.0, 0.0, grid=grid).data
    expected = float(np.float32(1e-10)) * phi / (0.01**2 * arcsec2rad**2)
    np.testing.assert_allclose(
        radiance, expected, rtol=roundoff(1, n_ops=32, dtype=radiance.dtype), atol=0
    )
    path = make_path(n=20, step=0.3, cube=True)
    quadrature = Quadrature.build((Band(549.5, 550.5),), scene, path, order=1)
    rates = photon_rates(scene, path, exposure_for(scene, quadrature, roll_deg=30))
    # Absolute incoming local-zodi normalization, on an interior pixel.
    lz = scene.local_zodi.photon_radiance(550.0, scene.sky_frame, 0.0)
    expected_z = (
        40
        * 0.4
        * 0.3
        * lz
        * float(path.detector.pixel_scale_arcsec) ** 2
        * arcsec2rad**2
    )
    np.testing.assert_allclose(
        rates.local_zodi[0, 0, 10, 10],
        expected_z,
        rtol=roundoff(1, n_ops=32, dtype=radiance.dtype),
        atol=0,
    )
    assert float(rates.exozodi.sum()) > 0
