"""A scene without a sky position renders with a pointing-independent local zodi."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_path, make_system, with_zodi
from optixstuff.spectral import Band
from skyscapes import AYOZodi, LeinertZodi

import coronagraphoto as cp


def _unplaced(scene):
    return eqx.tree_at(
        lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
        scene,
        (None, None),
        is_leaf=lambda x: x is None,
    )


def test_unplaced_scene_with_ayo_zodi_renders():
    """AYOZodi ignores the sightline, so no sky position is needed and none is used."""
    placed = with_zodi(make_system(), AYOZodi())
    scene = _unplaced(placed)
    assert scene.ra_deg is None
    path = make_path(n=8)
    bands = (Band(500.0, 600.0),)
    images = cp.expected(scene, path, cp.build_exposure(scene, path, bands, time_d=3.0))
    reference = cp.expected(
        placed, path, cp.build_exposure(placed, path, bands, time_d=3.0)
    )
    assert float(jnp.sum(images.local_zodi)) > 0
    np.testing.assert_array_equal(images.local_zodi, reference.local_zodi)


def test_unplaced_scene_with_leinert_zodi_names_the_fix():
    """A pointing-dependent model on an unplaced scene fails with a ValueError."""
    scene = _unplaced(with_zodi(make_system(), LeinertZodi()))
    path = make_path(n=8)
    exposure = cp.build_exposure(scene, path, (Band(500.0, 600.0),), time_d=0.0)
    with pytest.raises(ValueError, match=r"no sky position.*ra_deg"):
        cp.photon_rates(scene, path, exposure)
    placed = with_zodi(make_system(), LeinertZodi())
    assert cp.photon_rates(placed, path, exposure).local_zodi.shape[-1] == 8
