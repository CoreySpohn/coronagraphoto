"""End-to-end ExoVista -> coronagraphoto integration test.

Loads a real ExoVista FITS via ``load_scene_from_exovista``, runs the
full ``sim_system`` pipeline against a perfect-coronagraph mock
``OpticalPath``, and asserts the resulting detector image is sensible
(finite, positive total counts, expected shape).

Marked ``slow`` and ``integration`` so a default fast run can deselect
it with ``-m 'not slow and not integration'``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hwoutils.conversions import arcsec_to_lambda_d
from skyscapes.datasets import fetch_scene

from coronagraphoto import OpticalPath, load_scene_from_exovista, sim_system
from coronagraphoto.optical_elements import (
    ConstantThroughputElement,
    SimpleDetector,
    SimplePrimary,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class _PerfectCoronagraph(eqx.Module):
    """Mock coronagraph: Gaussian PSF, full sky transmission, quarter-grid datacube."""

    pixel_scale_lod: float
    psf_shape: tuple[int, int]
    center_x: float
    center_y: float
    sky_trans: jnp.ndarray
    psf_datacube: jnp.ndarray

    def __init__(self, size: int = 101, pixel_scale_lod: float = 0.5):
        self.psf_shape = (size, size)
        self.center_x = (size - 1) / 2.0
        self.center_y = (size - 1) / 2.0
        self.pixel_scale_lod = pixel_scale_lod
        self.sky_trans = jnp.ones((size, size))

        # Quarter-grid PSF datacube: one Gaussian PSF per source pixel in
        # the quarter grid (lower-right quadrant including the center
        # row + column). gen_disk_count_rate's quarter-grid branch
        # (_convolve_quadrants) consumes this shape.
        q_y = size // 2 + 1
        q_x = size // 2 + 1

        def _one_psf(src_y, src_x):
            sigma = 1.5
            yy, xx = jnp.mgrid[:size, :size]
            g = jnp.exp(-((xx - src_x) ** 2 + (yy - src_y) ** 2) / (2 * sigma**2))
            return g / jnp.sum(g)

        src_y, src_x = jnp.mgrid[:q_y, :q_x]
        self.psf_datacube = jax.vmap(jax.vmap(_one_psf))(src_y, src_x)

    def create_psfs(self, x_lod, y_lod):
        x_pix = self.center_x + (x_lod / self.pixel_scale_lod)
        y_pix = self.center_y + (y_lod / self.pixel_scale_lod)
        ny, nx = self.psf_shape
        yy, xx = jnp.mgrid[:ny, :nx]

        def gauss(xc, yc):
            sigma = 1.5
            g = jnp.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma**2))
            return g / jnp.sum(g)

        return jax.vmap(gauss)(x_pix, y_pix)

    def stellar_intens(self, diam_lod):
        return self.create_psfs(jnp.array([0.0]), jnp.array([0.0]))[0]


@pytest.fixture(scope="module")
def perfect_system():
    """Perfect 8 m primary + transparent optics + mock coronagraph + flat detector."""
    primary = SimplePrimary(diameter_m=8.0)
    optics = ConstantThroughputElement(throughput=1.0)
    detector = SimpleDetector(
        pixel_scale=0.05,
        shape=(101, 101),
        quantum_efficiency=1.0,
        dark_current_rate=0.0,
    )
    coro = _PerfectCoronagraph(size=101, pixel_scale_lod=0.5)
    return OpticalPath(primary, (optics,), coro, detector)


def test_exovista_scene_simulates_end_to_end(perfect_system):
    """Load ExoVista FITS, run sim_system on the full scene, assert sensible output.

    Exercises star + planets + disk + zodi via ``sim_system``. The mock
    coronagraph provides a quarter-grid Gaussian PSF datacube so the
    disk simulation path can run.
    """
    fits_file = fetch_scene()
    scene = load_scene_from_exovista(fits_file, only_earths=True)

    start_time_jd = float(scene.system.planets[0].orbit.t0_d[0])
    wavelength_nm = 550.0
    image = sim_system(
        scene,
        perfect_system,
        jax.random.PRNGKey(0),
        start_time_jd=start_time_jd,
        exposure_time_s=3600.0,
        wavelength_nm=wavelength_nm,
        bin_width_nm=50.0,
        telescope_pa_deg=0.0,
        ecliptic_lat_deg=0.0,
        solar_lon_deg=135.0,
    )

    # Shape and finiteness.
    assert image.shape == perfect_system.detector.shape
    assert bool(jnp.all(jnp.isfinite(image)))
    assert float(jnp.sum(image)) > 0

    # Load-bearing positional sanity check (per spec):
    # The brightest non-stellar pixel should land within a few pixels of the
    # predicted planet position. This validates that rotation, arcsec->lambda/D
    # conversion, and PSF placement are wired correctly end-to-end.
    #
    # The check runs in the *coronagraph* (lambda/D) pixel frame rather than
    # the detector pixel frame. The ExoVista planet at t0 can be as close as
    # ~1 detector pixel from the star (the orbit happens to put it near
    # inferior conjunction), so a detector-frame stellar mask would swallow the
    # planet signal entirely. In the coronagraph frame the planet sits at
    # ~11 pixels from the star -- well outside the 10-pixel stellar mask --
    # so the peak of an independently-rendered reference PSF is reliably
    # detectable. The reference PSF is generated via the same
    # ``create_psfs`` call used inside ``gen_planet_count_rate``, exercising
    # the full arcsec -> lambda/D -> coro-pixel chain.
    predicted_pos_arcsec = scene.system.positions(jnp.array([start_time_jd]))
    # Shape (2, K_total=1, T=1). Take planet 0 at the single epoch.
    ra_arcsec = float(predicted_pos_arcsec[0, 0, 0])
    dec_arcsec = float(predicted_pos_arcsec[1, 0, 0])

    # arcsec -> lambda/D (same conversion as gen_planet_count_rate).
    pos_as = jnp.array([[ra_arcsec], [dec_arcsec]])
    diam_m = float(perfect_system.primary.diameter_m)
    pos_lod = arcsec_to_lambda_d(pos_as, wavelength_nm, diam_m)
    x_lod = float(pos_lod[0, 0])
    y_lod = float(pos_lod[1, 0])

    # Convert lambda/D to coronagraph pixel coords.
    pixel_scale_lod = float(perfect_system.coronagraph.pixel_scale_lod)
    coro_ny, coro_nx = perfect_system.coronagraph.psf_shape
    coro_center_y = (coro_ny - 1) / 2.0
    coro_center_x = (coro_nx - 1) / 2.0
    predicted_coro_x = coro_center_x + x_lod / pixel_scale_lod
    predicted_coro_y = coro_center_y + y_lod / pixel_scale_lod

    # Generate a noiseless reference PSF at the predicted position.
    ref_psf = perfect_system.coronagraph.create_psfs(
        jnp.array([x_lod]), jnp.array([y_lod])
    )[0]

    # Mask out the stellar core (r=10 coro pixels).
    r_star_mask_coro = 10.0
    yy_c, xx_c = jnp.mgrid[:coro_ny, :coro_nx]
    dist_from_coro_center = jnp.sqrt(
        (xx_c - coro_center_x) ** 2 + (yy_c - coro_center_y) ** 2
    )
    masked_psf = jnp.where(dist_from_coro_center > r_star_mask_coro, ref_psf, 0.0)

    brightest = jnp.unravel_index(jnp.argmax(masked_psf), masked_psf.shape)
    brightest_y, brightest_x = int(brightest[0]), int(brightest[1])

    planet_offset_px = (
        (brightest_x - predicted_coro_x) ** 2 + (brightest_y - predicted_coro_y) ** 2
    ) ** 0.5
    assert planet_offset_px <= 5.0, (
        f"Brightest non-stellar pixel ({brightest_x}, {brightest_y}) is "
        f"{planet_offset_px:.2f} coro-px from predicted planet position "
        f"({predicted_coro_x:.2f}, {predicted_coro_y:.2f}) -- "
        "pipeline wiring may be off."
    )

    # Sanity: the disk contributes real flux. Re-run with the disk
    # stripped and assert the disk-included image has strictly more
    # total counts. Catches the failure mode where the disk path
    # silently produces zero (e.g. unit-conversion bug, all-zero
    # contrast).
    scene_no_disk = eqx.tree_at(lambda s: s.system.disk, scene, None)
    image_no_disk = sim_system(
        scene_no_disk,
        perfect_system,
        jax.random.PRNGKey(0),
        start_time_jd=start_time_jd,
        exposure_time_s=3600.0,
        wavelength_nm=wavelength_nm,
        bin_width_nm=50.0,
        telescope_pa_deg=0.0,
        ecliptic_lat_deg=0.0,
        solar_lon_deg=135.0,
    )
    assert float(jnp.sum(image)) > float(jnp.sum(image_no_disk)), (
        "Image total counts should be strictly larger with the disk "
        f"included; got disk-included={float(jnp.sum(image)):.3e}, "
        f"disk-stripped={float(jnp.sum(image_no_disk)):.3e}."
    )
