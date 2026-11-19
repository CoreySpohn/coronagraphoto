"""The aperture record declares the scene paths it reads, and the declaration is exact.

The exposure declares none: no scene value is cached in it.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PlateauCoronagraph, make_path, make_system, with_zodi
from hwoutils.model_paths import paths_overlap
from optixstuff.spectral import Band
from zodi.jax import LeinertZodi

import coronagraphoto as cp

_BANDS = (Band(500.0, 600.0),)


def _instrument():
    path = make_path(detector_n=24, detector_step=0.3)
    return eqx.tree_at(lambda p: p.coronagraph, path, PlateauCoronagraph(24, 0.25))


def _dotted(key_path):
    out = ""
    for key in key_path:
        if isinstance(key, jax.tree_util.GetAttrKey):
            out += ("." if out else "") + key.name
        elif isinstance(key, jax.tree_util.SequenceKey):
            out += f"[{key.idx}]"
        else:
            raise AssertionError(f"unexpected key {key!r}")
    return out


def _changed(a, b):
    return any(
        not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def _floating_leaves(scene):
    """Yield ``(dotted path, moved scene)`` with each floating leaf moved in turn."""
    flat, treedef = jax.tree_util.tree_flatten_with_path(scene)
    for i, (key_path, leaf) in enumerate(flat):
        if not (
            isinstance(leaf, float)
            or (
                isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating)
            )
        ):
            continue
        leaves = [x for _, x in flat]
        leaves[i] = leaf * 1.07 + 0.013
        yield _dotted(key_path), jax.tree_util.tree_unflatten(treedef, leaves)


@pytest.mark.parametrize("zodi", [None, LeinertZodi()], ids=["no-zodi", "zodi"])
def test_no_scene_leaf_changes_the_built_exposure(zodi):
    """Fails if the build caches a scene value, which would need a declared read.

    Every floating leaf of the scene is moved in turn and the exposure rebuilt; none
    changes it. The build uses the scene only for the spectral breaks of its local
    zodi, a constant of the zodi model, so ``build_exposure`` declares no read.
    """
    scene = make_system(separation_arcsec=0.03)
    if zodi is not None:
        scene = with_zodi(scene, zodi)
    instrument = _instrument()
    base = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0)
    moved = list(_floating_leaves(scene))
    assert moved, "the scene should have floating leaves to move"
    for path, other in moved:
        rebuilt = cp.build_exposure(other, instrument, _BANDS, time_d=0.0)
        assert jax.tree.structure(rebuilt) == jax.tree.structure(base), path
        assert not _changed(base, rebuilt), path


@pytest.mark.medium(reason="one aperture record per scene leaf")
@pytest.mark.parametrize("zodi", [None, LeinertZodi()], ids=["no-zodi", "zodi"])
def test_aperture_record_declares_exactly_what_it_reads(zodi):
    """Fails if the record reads a leaf outside ``reads``, or ``reads`` is loose.

    Every floating leaf of the scene is moved in turn and the record rebuilt. A leaf the
    record depends on must lie under a declared path (a refusal that missed it would let
    a stale record stand), and every declared path must cover a leaf the record depends
    on. The planets and their orbits are not read: a planet enters the read only as an
    offset and a contrast.
    """
    scene = make_system(separation_arcsec=0.03)
    if zodi is not None:
        scene = with_zodi(scene, zodi)
    instrument = _instrument()
    exposure = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0)
    response = cp.aperture_response(
        instrument,
        exposure.quadrature,
        n_separations=3,
        n_angles=3,
        max_separation_arcsec=0.04,
    )
    base = cp.aperture_record(scene, instrument, exposure, response)
    sensitive = [
        path
        for path, moved in _floating_leaves(scene)
        if _changed(base, cp.aperture_record(moved, instrument, exposure, response))
    ]
    assert len(jax.tree.leaves(base)) == len(jax.tree.leaves(replace(base, reads=())))
    assert jax.tree.structure(base) != jax.tree.structure(replace(base, reads=()))
    declared = base.reads
    assert sensitive, "the scene should have at least one leaf the record reads"
    for path in sensitive:
        assert any(paths_overlap(path, d) for d in declared), (path, declared)
    # The record stores the local-zodi model itself, which can have no array leaf to
    # move, so that path is checked by identity.
    assert base.local_zodi is scene.local_zodi
    assert (base.local_zodi is not None) == ("local_zodi" in declared)
    for d in declared:
        if d != "local_zodi":
            assert any(paths_overlap(path, d) for path in sensitive), (d, sensitive)
    assert not any(paths_overlap(p, "dynamics.orbits") for p in sensitive)
    assert not any(paths_overlap(p, "dynamics.planets") for p in sensitive)
    assert declared == (
        "star",
        "dynamics.frame",
        *(() if zodi is None else ("local_zodi",)),
    )
