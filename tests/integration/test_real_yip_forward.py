"""End-to-end ExoVista -> coronagraphoto integration test against a REAL yippy YIP.

Mirrors ``test_exovista_forward.py`` but swaps the ``_PerfectCoronagraph``
mock for a real ``yippy.EqxCoronagraph`` loaded from the eac1_aavc_512
YIP fetched via ``coronagraphoto.datasets.fetch_coronagraph``. Validates
that ``sim_system`` runs end-to-end against actual coronagraph data, not
just our test mocks.

Marked ``slow`` and ``integration`` so default fast runs deselect with
``-m 'not slow and not integration'``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from skyscapes.datasets import fetch_scene
from yippy import EqxCoronagraph

from coronagraphoto import load_scene_from_exovista, sim_system
from coronagraphoto.core.optical_path import OpticalPath
from coronagraphoto.datasets import fetch_coronagraph
from coronagraphoto.optical_elements import (
    ConstantThroughputElement,
    PrimaryAperture,
    SimpleDetector,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class _SkyTransAdapter(eqx.Module):
    """Wraps a yippy.EqxCoronagraph so ``sky_trans`` is callable.

    yippy.EqxCoronagraph stores ``sky_trans`` as a plain Array attribute;
    coronagraphoto's ``gen_disk_count_rate`` / ``gen_zodi_count_rate``
    call ``coronagraph.sky_trans()`` as a method (matching the test-mock
    convention). This wrapper bridges the two until the two libraries
    converge on one calling convention.

    Inherits the rest of the EqxCoronagraph interface via attribute
    forwarding to ``self.inner``.
    """

    inner: EqxCoronagraph

    @property
    def pixel_scale_lod(self) -> float:
        return self.inner.pixel_scale_lod

    @property
    def psf_shape(self) -> tuple[int, int]:
        return self.inner.psf_shape

    @property
    def psf_datacube(self):
        return self.inner.psf_datacube

    def create_psfs(self, x_lod, y_lod):
        return self.inner.create_psfs(x_lod, y_lod)

    def stellar_intens(self, diam_lod):
        return self.inner.stellar_intens(diam_lod)

    def sky_trans(self):
        return self.inner.sky_trans


@pytest.fixture(scope="module")
def real_optical_path():
    """OpticalPath with an 8 m primary + real yippy eac1_aavc_512 coronagraph."""
    yip_path = fetch_coronagraph()
    eqx_coro = EqxCoronagraph(yip_path, ensure_psf_datacube=True)
    coro = _SkyTransAdapter(inner=eqx_coro)

    ny, nx = coro.psf_shape
    primary = PrimaryAperture(diameter_m=8.0, obscuration_factor=0.0)
    optics = ConstantThroughputElement(throughput=1.0)
    detector = SimpleDetector(
        pixel_scale=0.05,
        shape=(ny, nx),
        quantum_efficiency=1.0,
        dark_current_rate=0.0,
    )
    return OpticalPath(primary, (optics,), coro, detector)


def test_exovista_scene_simulates_end_to_end_with_real_yip(real_optical_path):
    """Full sim_system pipeline against a real yippy YIP, not a mock."""
    fits_file = fetch_scene()
    scene = load_scene_from_exovista(fits_file, only_earths=True)

    start_time_jd = float(scene.system.planets[0].orbit.t0_d[0])
    wavelength_nm = 550.0
    image = sim_system(
        scene=scene,
        optical_path=real_optical_path,
        start_time_jd=start_time_jd,
        exposure_time_s=3600.0,
        wavelength_nm=wavelength_nm,
        bin_width_nm=50.0,
        telescope_pa_deg=0.0,
        prng_key=jax.random.PRNGKey(0),
    )

    # Shape and finiteness.
    assert image.shape == real_optical_path.detector.shape
    assert bool(jnp.all(jnp.isfinite(image)))
    assert float(jnp.sum(image)) > 0

    # Sanity: the disk contributes real flux. Strip and re-run to confirm.
    scene_no_disk = eqx.tree_at(lambda s: s.system.disk, scene, None)
    image_no_disk = sim_system(
        scene=scene_no_disk,
        optical_path=real_optical_path,
        start_time_jd=start_time_jd,
        exposure_time_s=3600.0,
        wavelength_nm=wavelength_nm,
        bin_width_nm=50.0,
        telescope_pa_deg=0.0,
        prng_key=jax.random.PRNGKey(0),
    )
    assert float(jnp.sum(image)) > float(jnp.sum(image_no_disk)), (
        "Image total counts should be strictly larger with the disk "
        f"included; got disk-included={float(jnp.sum(image)):.3e}, "
        f"disk-stripped={float(jnp.sum(image_no_disk)):.3e}."
    )
