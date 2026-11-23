"""The exposure and the aperture record declare the scene paths they read, exactly.

The exposure reads the knot grids of the scene's tables and caches no value; the
record also holds the star's node-effective flux.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PlateauCoronagraph, make_path, make_system, with_zodi
from hwoutils.model_paths import paths_overlap
from optixstuff.spectral import Band, Quadrature
from skyscapes import LeinertZodi, PrecomputedZodi
from skyscapes.scene import Star

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


_KNOTS = {
    "planets[0].physical_model.wavelengths_nm",
    "star._wavelengths_nm",
    "local_zodi.wavelengths_nm",
}


def _tabulated_scene(zodi):
    """A scene whose star, planet and local zodi (when tabulated) carry knots."""
    wave = jnp.array([400.0, 450.0, 520.0, 600.0, 700.0])
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.array([[1.0], [2.0], [3.0], [2.5], [2.0]]),
    )
    scene = make_system(star=star, separation_arcsec=0.03)
    return scene if zodi is None else with_zodi(scene, zodi)


_ZODIS = {
    "no-zodi": None,
    "leinert": LeinertZodi(),
    "table": PrecomputedZodi(jnp.array([450.0, 550.0, 650.0]), jnp.full(3, 1e12)),
}


@pytest.mark.parametrize("zodi", _ZODIS.values(), ids=_ZODIS.keys())
def test_exactly_the_knot_leaves_change_the_build(zodi):
    """Fails if the build reads a value leaf, or ``reads`` misses a knot leaf.

    Every floating leaf of the scene is moved in turn and the exposure rebuilt.
    The build must raise (the digest of a table no longer matches its knots)
    for the knot leaves and no other, those leaves must be exactly
    ``Exposure.reads``, and every other leaf must leave the built exposure
    unchanged, array for array. The knot leaves are named here by hand: the
    star's wavelength grid, the planet's, and the tabulated zodi's. A
    smooth local zodi has none.
    """
    scene = _tabulated_scene(zodi)
    instrument = _instrument()
    base = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0)
    expected = {
        path
        for path in _KNOTS
        if path != "local_zodi.wavelengths_nm" or isinstance(zodi, PrecomputedZodi)
    }
    assert set(base.reads) == expected
    moved = list(_floating_leaves(scene))
    assert len(moved) > len(expected), "the scene should have value leaves to move"
    refused = set()
    for path, other in moved:
        try:
            rebuilt = cp.build_exposure(other, instrument, _BANDS, time_d=0.0)
        except ValueError as error:
            assert "knot" in str(error), (path, error)
            refused.add(path)
            continue
        assert jax.tree.structure(rebuilt) == jax.tree.structure(base), path
        assert not _changed(base, rebuilt), path
    assert refused == expected


def _bolometric_scene():
    """A tabulated star over 400 to 700 nm, a planet table of 595 to 700 nm, a disk."""
    from skyscapes.disk import GraterDisk
    from zodi import SurfaceBrightnessDust

    star = Star(
        wavelengths_nm=jnp.linspace(400.0, 700.0, 7),
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.linspace(1.0, 2.0, 7)[:, None],
    )
    disk = GraterDisk(
        sma_AU=1.0,
        alpha_in=2.0,
        alpha_out=-2.0,
        ksi0_AU=0.05,
        gamma=2.0,
        beta=1.0,
        rmin_AU=0.1,
        rmax_AU=5.0,
        wavelengths_nm=jnp.array([500.0, 700.0]),
        g_HG_grid=jnp.zeros(2),
        Ag_grid=jnp.ones(2),
        amplitude=SurfaceBrightnessDust(level=3.0, luminosity_rule="bolometric"),
        n_slices_los=24,
    )
    return make_system(
        star=star,
        separation_arcsec=0.03,
        disk=disk,
        wavelengths_nm=(595.0, 650.0, 700.0),
    )


def test_a_calibration_band_beyond_the_other_tables_is_resolved_for_the_star_only():
    """Fails if a calibration band is paired with a table that does not cover it.

    A bolometric-rule disk averages the host star over a V band of 506 to 594
    nm. The planet's table here spans 595 to 700 nm and the exposure observes
    only inside it, so a rule for the planet on the V band would be refused for
    leaving the planet's table. The build must resolve the calibration band for
    the star's table alone: it raises nothing, the star's rule exists on the V
    quadrature, and the planet's rule does not.
    """
    scene = _bolometric_scene()
    exposure = cp.build_exposure(
        scene, _instrument(), (Band(610.0, 690.0),), time_d=0.0
    )
    (v_band,) = scene.calibration_bands()
    v_rule = Quadrature.build((v_band,))
    star, planet = scene.spectral_tables()
    assert exposure.resolution.rule(v_rule, star).kind in ("table", "identity")
    with pytest.raises(LookupError):
        exposure.resolution.rule(v_rule, planet)
    # The disk then renders: its level reads the calibration the sources hold.
    rates = cp.photon_rates(scene, _instrument(), exposure)
    assert np.all(np.isfinite(np.asarray(rates.exozodi)))
    assert float(jnp.sum(rates.exozodi)) > 0


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
