"""End-to-end ExoVista -> coronagraphoto integration test against a REAL yippy YIP.

Mirrors ``test_exovista_forward.py`` but swaps the ``_PerfectCoronagraph``
mock for a real ``yippy.EqxCoronagraph`` loaded from the eac1_aavc_512
YIP fetched via ``coronagraphoto.datasets.fetch_coronagraph``. Validates
that ``system_readout`` runs end-to-end against actual coronagraph data, not
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

from coronagraphoto import OpticalPath, load_scene_from_exovista, system_readout
from coronagraphoto.datasets import fetch_coronagraph
from coronagraphoto.optical_elements import (
    ConstantThroughputElement,
    SimpleDetector,
    SimplePrimary,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def real_optical_path():
    """OpticalPath with an 8 m primary + real yippy eac1_aavc_512 coronagraph.

    Skips PSF-datacube construction (``ensure_psf_datacube=False``)
    because this test exercises only the star + planet paths against
    the real YIP. The disk path -- the only consumer of the datacube
    -- is covered against the perfect-coronagraph mock in
    ``test_exovista_forward.py``, where an (n_src, n_src, npixels,
    npixels) datacube costs nothing to build. Precomputing the real
    datacube was making this test take 20+ minutes; without it the
    same coverage runs in seconds.
    """
    yip_path = fetch_coronagraph()
    coro = EqxCoronagraph(yip_path, ensure_psf_datacube=False)

    ny, nx = coro.psf_shape
    primary = SimplePrimary(diameter_m=8.0)
    optics = ConstantThroughputElement(throughput=1.0)
    detector = SimpleDetector(
        pixel_scale=0.05,
        shape=(ny, nx),
        quantum_efficiency=1.0,
        dark_current_rate=0.0,
    )
    return OpticalPath(primary, (optics,), coro, detector)


def test_exovista_scene_simulates_end_to_end_with_real_yip(real_optical_path):
    """Star + planet pipeline against a real yippy YIP, not a mock.

    Disk path is stripped because the real-yippy PSF datacube it would
    require is prohibitively expensive to build (see fixture docstring);
    disk coverage lives in ``test_exovista_forward.py`` against the
    mock.
    """
    fits_file = fetch_scene()
    scene = load_scene_from_exovista(fits_file, only_earths=True)
    scene = eqx.tree_at(lambda s: s.system.disk, scene, None)

    start_time_jd = float(scene.system.planets[0].orbit.t0_d[0])
    wavelength_nm = 550.0
    image = system_readout(
        scene,
        real_optical_path,
        jax.random.PRNGKey(0),
        start_time_jd=start_time_jd,
        exposure_time_s=3600.0,
        wavelength_nm=wavelength_nm,
        bin_width_nm=50.0,
        telescope_pa_deg=0.0,
        ecliptic_lat_deg=0.0,
        solar_lon_deg=135.0,
    )

    assert image.shape == real_optical_path.detector.shape
    assert bool(jnp.all(jnp.isfinite(image)))
    assert float(jnp.sum(image)) > 0
