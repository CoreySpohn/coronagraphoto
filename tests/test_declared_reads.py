"""Built objects declare the scene paths they read, and the declaration is exact."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PlateauCoronagraph, make_path, make_system, with_zodi
from optixstuff.spectral import Band
from orbix.system import paths_overlap
from skyscapes.scene import Scene
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


def _touched_scene_attributes(monkeypatch, call):
    """Run ``call`` and return the public Scene attribute names it read."""
    seen = set()
    original = Scene.__getattribute__

    def spy(self, name):
        if not name.startswith("_"):
            seen.add(name)
        return original(self, name)

    monkeypatch.setattr(Scene, "__getattribute__", spy)
    call()
    monkeypatch.undo()
    return seen


@pytest.mark.parametrize("zodi", [None, LeinertZodi()], ids=["no-zodi", "zodi"])
def test_build_exposure_declares_exactly_what_it_reads(monkeypatch, zodi):
    """Fails if the build reads a scene value its ``reads`` does not name.

    The build is host code, so a spy on the scene's attribute access sees everything it
    touches. It touches only ``local_zodi``, and uses its value only when there is one.
    """
    scene = make_system(separation_arcsec=0.03)
    if zodi is not None:
        scene = with_zodi(scene, zodi)
    instrument = _instrument()
    built = {}
    touched = _touched_scene_attributes(
        monkeypatch,
        lambda: built.update(
            exposure=cp.build_exposure(scene, instrument, _BANDS, time_d=0.0)
        ),
    )
    exposure = built["exposure"]
    assert touched == {"local_zodi"}
    assert exposure.reads == (() if zodi is None else ("local_zodi",))
    assert isinstance(exposure.reads, tuple)


def test_reads_are_static_metadata_not_array_leaves():
    """Fails if ``reads`` enters the pytree, where it would recompile or be traced."""
    scene = with_zodi(make_system(separation_arcsec=0.03), LeinertZodi())
    exposure = cp.build_exposure(scene, _instrument(), _BANDS, time_d=0.0)
    other = replace(exposure, reads=())
    assert len(jax.tree.leaves(exposure)) == len(jax.tree.leaves(other))
    assert all("local_zodi" not in str(leaf) for leaf in jax.tree.leaves(exposure))
    assert jax.tree.structure(exposure) != jax.tree.structure(other)


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
    flat, treedef = jax.tree_util.tree_flatten_with_path(scene)
    sensitive = []
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
        moved = jax.tree_util.tree_unflatten(treedef, leaves)
        if _changed(base, cp.aperture_record(moved, instrument, exposure, response)):
            sensitive.append(_dotted(key_path))
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
