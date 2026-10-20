"""Sky-to-detector conventions through the Gaussian table optical path."""

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from hwoutils.conversions import arcsec_to_lambda_d
from optixstuff import ConstantThroughput, IdealDetector, OpticalPath, SimplePrimary
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit
from skyscapes.disk import GraterDisk
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import Planet, Star

from coronagraphoto.simulation import disk_rate, planet_rate
from tests.integration.test_exovista_forward import _PerfectCoronagraph

_OBS = dict(start_time_jd=0.0, wavelength_nm=550.0, bin_width_nm=10.0)
_SIZE = 65
_CENTER = (_SIZE - 1) // 2
_SCALE = 0.01
_SEPARATION_PIX = 20.0  # 2 AU at 10 pc, sampled at 0.01 arcsec/pixel.


@pytest.fixture(scope="module")
def orientation_setup():
    """Small version of the existing Gaussian datacube path; no fetched data."""
    primary = SimplePrimary(diameter_m=2.0)
    coro = _PerfectCoronagraph(
        size=_SIZE,
        pixel_scale_lod=arcsec_to_lambda_d(_SCALE, 550.0, primary.diameter_m),
    )
    optical_path = OpticalPath(
        primary,
        (ConstantThroughput(throughput=1.0),),
        coro,
        IdealDetector(pixel_scale_arcsec=_SCALE, shape=(_SIZE, _SIZE)),
    )
    star = Star(
        Ms_kg=Msun2kg,
        dist_pc=10.0,
        wavelengths_nm=jnp.array([400.0, 500.0, 600.0, 700.0]),
        times_jd=jnp.array([0.0, 1.0]),
        flux_density_jy=jnp.ones((4, 2)),
        diameter_arcsec=0.0,
    )
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    return optical_path, star, solver


def _planet_image(setup, sky_pa_deg, telescope_pa_deg):
    """Render a circular planet at its ascending node, whose sky PA is Omega."""
    optical_path, star, solver = setup
    orbit = KeplerianOrbit(
        a_AU=jnp.array([2.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.deg2rad(jnp.array([sky_pa_deg])),
        i_rad=jnp.deg2rad(jnp.array([75.0])),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        t0_d=jnp.array([0.0]),
    )
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=orbit,
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    # Verify the upstream sky position independently of detector placement.
    pa = np.deg2rad(sky_pa_deg)
    np.testing.assert_allclose(
        planet.position_arcsec(solver, jnp.array([0.0]), star=star)[:, 0, 0],
        0.2 * np.array([np.sin(pa), np.cos(pa)]),
        atol=1e-7,
    )
    return np.asarray(
        planet_rate(
            planet,
            optical_path,
            star=star,
            trig_solver=solver,
            telescope_pa_deg=telescope_pa_deg,
            **_OBS,
        )
    )


def _peak_offset(image):
    """Return the PSF peak's (row, column) offset from the detector center."""
    assert np.isfinite(image).all() and image.max() > 0
    return np.array(np.unravel_index(image.argmax(), image.shape)) - _CENTER


def test_east_is_left_at_zero_roll(orientation_setup):
    """A positive Delta alpha* places the planet below the center column."""
    offset = _peak_offset(_planet_image(orientation_setup, 90.0, 0.0))
    assert offset[1] < 0
    np.testing.assert_allclose(offset, [0, -_SEPARATION_PIX], atol=1.0)


def test_quarter_turn_maps_east_up_and_north_right(orientation_setup):
    """At telescope PA 90, detector +y is east and detector +x is north."""
    east = _peak_offset(_planet_image(orientation_setup, 90.0, 90.0))
    north = _peak_offset(_planet_image(orientation_setup, 0.0, 90.0))
    np.testing.assert_allclose(east, [_SEPARATION_PIX, 0], atol=1.0)
    np.testing.assert_allclose(north, [0, _SEPARATION_PIX], atol=1.0)


def test_telescope_pa_is_sky_pa_of_detector_y(orientation_setup):
    """A source at sky PA 30 lies on detector +y when telescope PA is 30."""
    offset = _peak_offset(_planet_image(orientation_setup, 30.0, 30.0))
    np.testing.assert_allclose(offset, [_SEPARATION_PIX, 0], atol=1.0)


@pytest.mark.parametrize("telescope_pa_deg", [0.0, 30.0])
def test_planet_stays_on_disk_ansa(orientation_setup, telescope_pa_deg):
    """An inclined narrow ring and a planet on its ansa share detector pixels."""
    optical_path, star, _ = orientation_setup
    disk = GraterDisk(
        sma_AU=2.0,
        alpha_in=12.0,
        alpha_out=-12.0,
        ksi0_AU=0.03,
        gamma=2.0,
        beta=0.0,
        rmin_AU=1.85,
        rmax_AU=2.15,
        wavelengths_nm=jnp.array([400.0, 500.0, 600.0, 700.0]),
        g_HG_grid=jnp.zeros(4),
        Ag_grid=jnp.ones(4),
        nx=_SIZE,
        ny=_SIZE,
        pixel_scale_arcsec=_SCALE,
        dist_pc=10.0,
        n_slices_los=64,
    )
    disk_image = np.asarray(
        disk_rate(
            disk,
            optical_path,
            star=star,
            incl_deg=jnp.asarray(75.0),
            # Disk PA runs east toward north; sky PA 60 is disk PA 30.
            pa_deg=30.0,
            telescope_pa_deg=telescope_pa_deg,
            **_OBS,
        )
    )
    planet_image = _planet_image(orientation_setup, 60.0, telescope_pa_deg)
    row, col = np.unravel_index(planet_image.argmax(), planet_image.shape)
    assert np.isfinite(disk_image).all() and disk_image.max() > 0
    assert disk_image[row, col] > 0.5 * disk_image.max()
    assert disk_image[row, col] > 10.0 * disk_image[row, 2 * _CENTER - col]
