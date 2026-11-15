"""Sky-to-detector conventions through the Gaussian table optical path."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.conversions import arcsec_to_lambda_d
from hwoutils.epoch import Epoch
from optixstuff import ConstantThroughput, IdealDetector, OpticalPath, SimplePrimary
from skyscapes import Scene
from skyscapes.disk import GraterDisk
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import BlackbodyStar, FlatStar, Planet, Star, System

from coronagraphoto.simulation import disk_rate, planet_rate, system_rate
from tests.integration.test_exovista_forward import _PerfectCoronagraph

_EPOCH = Epoch(2451545.0)
_OBS = dict(start_time_d=0.0, wavelength_nm=550.0, bin_width_nm=10.0)
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
        wavelengths_nm=jnp.array([400.0, 500.0, 600.0, 700.0]),
        times_d=jnp.array([0.0, 1.0]),
        flux_density_jy=jnp.ones((4, 2)),
        diameter_arcsec=0.0,
    )
    return optical_path, star


def _planet_image(setup, sky_pa_deg, telescope_pa_deg):
    """Render a circular planet at its ascending node, whose sky PA is Omega."""
    optical_path, star = setup

    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )
    system = System.from_classical(
        star,
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=Mearth2kg / Msun2kg,
        dist_pc=10.0,
        epoch=Epoch(2451545.0),
        a_AU=jnp.array([2.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.deg2rad(jnp.array([sky_pa_deg])),
        i_rad=jnp.deg2rad(jnp.array([75.0])),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        epoch_d=jnp.array([0.0]),
    )
    # Verify the upstream sky position independently of detector placement.
    pa = np.deg2rad(sky_pa_deg)
    np.testing.assert_allclose(
        system.positions(jnp.array([0.0]))[:, 0, 0],
        0.2 * np.array([np.sin(pa), np.cos(pa)]),
        atol=1e-7,
    )
    return np.asarray(
        planet_rate(
            system,
            optical_path,
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
    optical_path, star = orientation_setup
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
        n_slices_los=64,
    )
    disk_image = np.asarray(
        disk_rate(
            disk,
            optical_path,
            star=star,
            dist_pc=10.0,
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


def _system(star, planets, *, epoch=_EPOCH, epoch_d=0.0, a_AU=1.0):
    """Build small scenes with explicit reference epochs and nonzero masses."""
    return System.from_classical(
        star,
        planets,
        star_mass_Msun=1.0,
        planet_mass_Msun=Mearth2kg / Msun2kg,
        dist_pc=10.0,
        epoch=epoch,
        epoch_d=epoch_d,
        a_AU=a_AU,
        e=0.15,
        i_rad=0.7,
        W_rad=0.4,
        w_rad=0.2,
        M0_rad=0.3,
    )


def test_planet_tuple_batching_preserves_image(orientation_setup):
    """Fails if tuple indices replace flattened scene slices or bodies reorder.

    No existing image test compares split and combined photometric batches.
    """
    with jax.enable_x64():
        path, _ = orientation_setup
        star = FlatStar(flux_phot_per_nm_m2=1e9)
        radii, albedos = jnp.array([1.8, 0.7]), jnp.array([0.15, 0.6])
        split = tuple(
            Planet(radii[k : k + 1], LambertianPhysicalModel(Ag=albedos[k : k + 1]))
            for k in range(2)
        )
        combined = (Planet(radii, LambertianPhysicalModel(Ag=albedos)),)
        # Outer first exercises scene-to-chain order as well as tuple slicing.
        scenes = [
            Scene(_system(star, planets, a_AU=jnp.array([2.0, 1.0])), zodi=None)
            for planets in (split, combined)
        ]
        render = eqx.filter_jit(system_rate)
        args = dict(
            start_time_d=jnp.asarray(53.0),
            wavelength_nm=550.0,
            bin_width_nm=10.0,
            telescope_pa_deg=27.0,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=135.0,
            trace=True,
        )
        first, first_parts = render(scenes[0], path, **args)
        second, second_parts = render(scenes[1], path, **args)
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            first_parts["planets"], second_parts["planets"], rtol=1e-12, atol=0
        )
        assert first_parts["planets"].sum() > 0


class _DiameterCoronagraph(_PerfectCoronagraph):
    """Stellar leakage sensitive to resolved diameter, unlike the Gaussian mock."""

    def stellar_intens(self, diam_lod):
        return super().stellar_intens(diam_lod) * diam_lod**2


def test_physical_star_diameter_method_reaches_image():
    """Fails if a bound diameter method reaches angular conversion or is ignored.

    Existing star radiometry uses apparent point stars, so misses this path.
    """
    with jax.enable_x64():
        path = OpticalPath(
            SimplePrimary(2.0),
            (ConstantThroughput(throughput=1.0),),
            _DiameterCoronagraph(size=9),
            IdealDetector(pixel_scale_arcsec=0.05, shape=(9, 9)),
        )
        star = BlackbodyStar(teff_K=5772.0, radius_m=6.957e8)
        render = eqx.filter_jit(system_rate)
        args = dict(
            start_time_d=jnp.asarray(0.0),
            wavelength_nm=550.0,
            bin_width_nm=10.0,
            telescope_pa_deg=0.0,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=135.0,
        )
        images = []
        for distance in (10.0, 20.0):
            physical = _system(star, ())
            physical = eqx.tree_at(
                lambda s: s.dynamics.frame.dist_pc, physical, distance
            )
            apparent_star = FlatStar(
                flux_phot_per_nm_m2=star.spec_flux_density(
                    550.0, 0.0, dist_pc=distance
                ),
                diameter_arcsec=star.diameter_arcsec(dist_pc=distance),
            )
            apparent = eqx.tree_at(lambda s: s.star, physical, apparent_star)
            image = render(Scene(physical, zodi=None), path, **args)
            expected = render(Scene(apparent, zodi=None), path, **args)
            np.testing.assert_allclose(image, expected, rtol=1e-12, atol=0)
            assert image.sum() > 0
            images.append(image)
        # Flux and diameter-squared leakage each fall as distance squared.
        np.testing.assert_allclose(images[0], 16 * images[1], rtol=1e-12, atol=0)


def test_image_same_absolute_date_with_different_scene_epochs(orientation_setup):
    """Fails if absolute dates enter propagation or the speckle/zodi clocks shift.

    Existing time tests vary observation times within a single reference epoch.
    """
    from orbix.observatory import ObservatoryL2Halo
    from skyscapes.background import LeinertZodi

    from tests.test_speckle import _MockSpeckle

    with jax.enable_x64():
        path, _ = orientation_setup
        orbit = ObservatoryL2Halo.from_default(equinox_mjd=51544.5)
        epochs = (Epoch(2451545.0, 0.125), Epoch(2451547.0, 0.375))
        absolute_start = Epoch(2451546.0, 0.875)
        images = []

        @eqx.filter_jit
        def render(scene, path, time_d):
            # The observatory requires MJD; reduce split epoch on the host.
            epoch = scene.system.epoch
            offset = (epoch.jd1 - 2400000.5) + epoch.jd2
            mjd = time_d + offset
            lat = orbit.ecliptic_latitude_deg(mjd, 1.0, 0.4)
            lon = orbit.helio_ecliptic_longitude_deg(mjd, 1.0, 0.4)
            return system_rate(
                scene,
                path,
                start_time_d=time_d,
                wavelength_nm=550.0,
                bin_width_nm=10.0,
                telescope_pa_deg=19.0,
                ecliptic_lat_deg=lat,
                solar_lon_deg=lon,
                trace=True,
            )

        for epoch in epochs:
            shift = epoch.days_between(epochs[0])
            star = Star(
                wavelengths_nm=jnp.array([400.0, 500.0, 600.0, 700.0]),
                times_d=jnp.array([-2.0, 4.0]) - shift,
                flux_density_jy=jnp.array([[1.0, 3.0]] * 4),
                diameter_arcsec=0.0,
            )
            planet = Planet(
                jnp.array([1.0]), LambertianPhysicalModel(Ag=jnp.array([0.3]))
            )
            system = _system(star, (planet,), epoch=epoch, epoch_d=-shift)
            scene = Scene(system, zodi=LeinertZodi())
            timed_path = eqx.tree_at(
                lambda p: p.speckle,
                path,
                _MockSpeckle(epoch_d=-shift),
                is_leaf=lambda x: x is None,
            )
            images.append(
                render(scene, timed_path, jnp.asarray(epoch.days(absolute_start)))
            )
        np.testing.assert_allclose(images[0][0], images[1][0], rtol=1e-10, atol=0)
        for source in images[0][1]:
            np.testing.assert_allclose(
                images[0][1][source], images[1][1][source], rtol=1e-10, atol=0
            )
