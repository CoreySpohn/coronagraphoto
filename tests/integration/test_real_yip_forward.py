"""Full FITS scene/YIP observation integration using small synthetic inputs."""

import equinox as eqx
import jax
import numpy as np
import pytest
from conftest import make_path, write_scene, write_yip
from hwoutils.grids import SkyOrientation
from hwoutils.spectral import Band
from optixstuff import FramePlan
from yippy import Coronagraph

from coronagraphoto import image_electrons, load_scene_from_exovista, observe


@pytest.mark.medium(reason="FITS scene and YIP loaders plus observation compilation")
def test_fits_scene_yip_observation(tmp_path):
    """Fails on incompatible scene, YIP, detector or observation adapters.

    Both loaders consume genuine FITS products. Deterministic rates must be
    positive; draw equality is bitwise because identical keys call one sampler.
    """
    scene = load_scene_from_exovista(
        write_scene(tmp_path / "scene.fits"), load_disk=False
    )
    coro = Coronagraph(write_yip(tmp_path / "yip"))
    path = eqx.tree_at(lambda p: p.coronagraph, make_path(n=24, step=0.5), coro)
    bands = (Band(545.0, 555.0),)
    plan = FramePlan(10.0, 1.0)
    options = dict(
        time_d=0.0, pointing=SkyOrientation(0), quadrature_order=1, zodi=scene.zodi
    )
    rates = image_electrons(scene.system, path, bands, **options)
    assert float(rates.planet.sum()) > 0
    assert float(rates.local_zodi.sum()) > 0
    key = jax.random.PRNGKey(0)
    image = observe(key, scene.system, path, bands, plan, **options)
    assert image.data.shape == (1, 24, 24)
    np.testing.assert_array_equal(image.data, path.detector.draw(key, rates, plan))
