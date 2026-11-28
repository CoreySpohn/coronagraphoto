"""The sampling reads only static knots; the aperture record declares what it reads.

The sampling reads the static knot grids of the scene's tables and caches no
value, so moving any floating scene leaf leaves it unchanged; it names the
frozen node rule and the tabulated scene tables it reduced in ``reads``. The
aperture record also holds the star's node-effective flux and names the scene
paths it holds fixed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    PlateauCoronagraph,
    make_path,
    make_system,
    with_instrument,
    with_zodi,
)
from hwoutils.model_paths import paths_overlap
from hwoutils.provenance import StampMismatchError
from optixstuff.sampling import SpectralSampling
from optixstuff.spectral import Band, Quadrature
from skyscapes import LeinertZodi, PrecomputedZodi
from skyscapes.scene import Scene, Star

import coronagraphoto as cp

_BANDS = (Band(500.0, 600.0),)


def _instrument():
    path = make_path(detector_n=24, detector_step=0.3)
    return eqx.tree_at(
        lambda p: p.instruments[0].coronagraph, path, PlateauCoronagraph(24, 0.25)
    )


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
        # NumPy leaves count: a table may keep host knots as given.
        if not (isinstance(leaf, float) or eqx.is_inexact_array(leaf)):
            continue
        leaves = [x for _, x in flat]
        leaves[i] = leaf * 1.07 + 0.013
        yield _dotted(key_path), jax.tree_util.tree_unflatten(treedef, leaves)


def _tabulated_scene(zodi, last_knot_nm=700.0, smooth_planet=False):
    """A scene whose star, planet and local zodi (when tabulated) carry knots."""
    wave = jnp.array([400.0, 450.0, 520.0, 600.0, last_knot_nm])
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.array([[1.0], [2.0], [3.0], [2.5], [2.0]]),
    )
    scene = make_system(star=star, separation_arcsec=0.03, smooth_planet=smooth_planet)
    return scene if zodi is None else with_zodi(scene, zodi)


_ZODIS = {
    "no-zodi": None,
    "leinert": LeinertZodi(),
    "table": PrecomputedZodi(jnp.array([450.0, 550.0, 650.0]), jnp.full(3, 1e12)),
}


def _bitwise_equal(a, b):
    """True when two trees have one structure and bitwise equal, same-dtype leaves."""
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        np.asarray(x).dtype == np.asarray(y).dtype
        and np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


@pytest.mark.parametrize("zodi", _ZODIS.values(), ids=_ZODIS.keys())
def test_sampling_of_a_round_trip_scene_equals_the_live_sampling(zodi):
    """Fails if the build reads anything the batch round trip changes.

    The scene goes through ``Scene.from_scenes`` and ``to_scenes`` (which
    return JAX leaves) and both are built into a sampling on the same
    instrument and bands. The two samplings must have one tree structure (the
    static knots and resolution keys included) and bitwise equal leaves of
    equal dtype: the same inputs go through the same host build, so there is no
    tolerance, and the declared reads are the same. Then every floating leaf of
    the scene is moved in turn and the sampling rebuilt: the build must accept
    the moved scene and return a sampling bitwise equal to the first, which is
    the evidence that the sampling reads only static knots.
    """
    scene = _tabulated_scene(None)
    (round_trip,) = Scene.from_scenes([scene]).to_scenes()
    if zodi is not None:  # a batch holds no local zodi; it is put back after
        scene, round_trip = with_zodi(scene, zodi), with_zodi(round_trip, zodi)
    instrument = _instrument()
    live = cp.build_exposure(_BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0)
    live_sampling = cp.build_sampling(instrument, live, scene=scene)
    other = cp.build_exposure(
        _BANDS, epoch=round_trip.epoch, instrument="imager", time_d=0.0
    )
    other_sampling = cp.build_sampling(instrument, other, scene=round_trip)
    assert live_sampling.reads == other_sampling.reads
    assert _bitwise_equal(live_sampling, other_sampling)
    for path, moved in _floating_leaves(scene):
        rebuilt = cp.build_exposure(
            _BANDS, epoch=moved.epoch, instrument="imager", time_d=0.0
        )
        rebuilt_sampling = cp.build_sampling(instrument, rebuilt, scene=moved)
        assert _bitwise_equal(live_sampling, rebuilt_sampling), path


@pytest.mark.parametrize(
    "value",
    [np.asarray(12.5), np.float64(12.5)],
    ids=["ndarray", "np.float64"],
)
def test_a_numpy_leaf_is_refused_by_the_sampling_build_with_its_path(value):
    """Fails if a NumPy leaf put into a scene by ``tree_at`` reaches the sampling build.

    ``eqx.tree_at`` bypasses the field converters, so a NumPy value can stand
    in a scene whose constructors never produced one. The build must raise
    ``TypeError`` naming the leaf's path, on a NumPy array and on a NumPy
    scalar; the unchanged scene builds. The truth is the path of the leaf that
    was replaced, so there is no tolerance.
    """
    scene = _tabulated_scene(None)
    instrument = _instrument()
    exposure = cp.build_exposure(
        _BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    cp.build_sampling(instrument, exposure, scene=scene)
    broken = eqx.tree_at(lambda s: s.dynamics.frame.dist_pc, scene, value)
    with pytest.raises(TypeError, match=r"\.dynamics\.frame\.dist_pc"):
        cp.build_sampling(instrument, exposure, scene=broken)


class _UserBox(eqx.Module):
    """A user's own Module, defined outside the owned packages."""

    inner: object


def test_a_numpy_leaf_in_a_user_module_is_refused_at_every_entry_with_its_path():
    """Fails if a NumPy leaf inside a user's Module nested in the scene gets through.

    The user Module is put into the scene by ``tree_at``. ``build_sampling``,
    ``photon_rates`` and ``expected`` must raise ``TypeError`` naming the path
    through the user Module (the walk enters it, as ``require_tree`` does), and
    ``expected`` under ``jit`` must run on the unchanged scene, where every leaf is
    a tracer. The truth is the replaced path, so there is no tolerance.
    """
    scene = _tabulated_scene(None, smooth_planet=True)
    instrument = _instrument()
    exposure = cp.build_exposure(
        _BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(instrument, exposure, scene=scene)
    broken = eqx.tree_at(
        lambda s: s.dynamics.frame.dist_pc, scene, _UserBox(np.asarray(12.5))
    )
    match = r"\.dynamics\.frame\.dist_pc.*\.inner"
    with pytest.raises(TypeError, match=match):
        cp.build_sampling(instrument, exposure, scene=broken)
    with pytest.raises(TypeError, match=match):
        cp.photon_rates(broken, instrument, exposure, sampling=sampling)
    with pytest.raises(TypeError, match=match):
        cp.expected(broken, instrument, exposure, sampling=sampling)
    rates = jax.jit(
        lambda sc: cp.expected(sc, instrument, exposure, sampling=sampling)
    )(scene)
    assert np.all(np.isfinite(np.asarray(rates.star_leakage)))


def test_replacing_a_stars_knots_leaves_the_old_sampling_refused():
    """Fails if a sampling built for one knot grid is accepted for another.

    The star is replaced by one whose static ``Knots`` has the same length and
    another last knot, on a scene whose planet has no table (so the star's is the
    only operator and no other refusal can stand in for it). Every leaf shape is
    unchanged and only the digest differs. The exposure built before the change
    must refuse the new scene (``StampMismatchError``: the sampling's stamp holds
    the old scene layout) in ``photon_rates``, and the sampling built after the
    change must render it. The truth is the presence of the layout stamp, so
    there is no tolerance.
    """
    scene = _tabulated_scene(None, smooth_planet=True)
    instrument = _instrument()
    exposure = cp.build_exposure(
        _BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(instrument, exposure, scene=scene)
    moved = eqx.tree_at(
        lambda s: s.star, scene, _tabulated_scene(None, 710.0, smooth_planet=True).star
    )
    assert moved.star.knots.digest != scene.star.knots.digest
    with pytest.raises(StampMismatchError):
        cp.photon_rates(moved, instrument, exposure, sampling=sampling)
    fresh = cp.build_exposure(
        _BANDS, epoch=moved.epoch, instrument="imager", time_d=0.0
    )
    fresh_sampling = cp.build_sampling(instrument, fresh, scene=moved)
    assert np.all(
        np.isfinite(
            np.asarray(
                cp.photon_rates(
                    moved, instrument, fresh, sampling=fresh_sampling
                ).star_leakage
            )
        )
    )


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
        (Band(610.0, 690.0),), epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(_instrument(), exposure, scene=scene)
    (v_band,) = scene.calibration_bands()
    v_rule = Quadrature.build((v_band,))
    star, planet = scene.spectral_tables()
    assert sampling.resolution.rule(v_rule, star).kind in ("table", "identity")
    with pytest.raises(LookupError):
        sampling.resolution.rule(v_rule, planet)
    # The disk then renders: its level reads the calibration the sources hold.
    rates = cp.photon_rates(scene, _instrument(), exposure, sampling=sampling)
    assert np.all(np.isfinite(np.asarray(rates.exozodi)))
    assert float(jnp.sum(rates.exozodi)) > 0


@pytest.mark.medium(reason="one aperture record per scene leaf")
@pytest.mark.parametrize("host", [False, True], ids=["jax-input", "numpy-input"])
@pytest.mark.parametrize("zodi", [None, LeinertZodi()], ids=["no-zodi", "zodi"])
def test_aperture_record_declares_exactly_what_it_reads(zodi, host):
    """Fails if the record reads a leaf outside ``reads``, or ``reads`` is loose.

    Every floating leaf of the scene is moved in turn and the record rebuilt. A leaf the
    record depends on must lie under a declared path (a refusal that missed it would let
    a stale record stand), and every declared path must cover a leaf the record depends
    on. The planets and their orbits are not read: a planet enters the read only as an
    offset and a contrast. With ``host`` the planet's table, the star's table and
    the orbit arrays are built from NumPy arrays, which the constructors convert,
    so the declaration is the same as for JAX input.
    """
    scene = make_system(separation_arcsec=0.03, host=host)
    if zodi is not None:
        scene = with_zodi(scene, zodi)
    instrument = _instrument()
    exposure = cp.build_exposure(
        _BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(instrument, exposure, scene=scene)
    response = cp.aperture_response(
        instrument,
        sampling.quadrature,
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
    assert len(jax.tree.leaves(base)) == len(
        jax.tree.leaves(with_instrument(base, reads=()))
    )
    assert jax.tree.structure(base) != jax.tree.structure(
        with_instrument(base, reads=())
    )
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


def test_sampling_reads_follow_the_tables_the_build_reduced():
    """Fails if ``reads`` names a path the build did not freeze, or misses one.

    Three samplings: a smooth spectrum (no operator, so nothing frozen and no
    reads), a tabulated star (the frozen node rule and the star's table are
    declared) and a hand-built sampling with a non-empty resolution (the
    caller's tuple stands). The truth is the set of paths the build reduced,
    so the comparison is exact.
    """
    instrument = _instrument()
    smooth = make_system(smooth_planet=True)
    smooth_event = cp.build_exposure(
        _BANDS, epoch=smooth.epoch, instrument="imager", time_d=0.0
    )
    smooth_sampling = cp.build_sampling(instrument, smooth_event, scene=smooth)
    assert not smooth_sampling.resolution.keys
    assert smooth_sampling.reads == ()

    tabulated = _tabulated_scene(None, smooth_planet=True)
    event = cp.build_exposure(
        _BANDS, epoch=tabulated.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(instrument, event, scene=tabulated)
    assert sampling.resolution.keys
    assert "sampling.quadrature.nodes_nm" in sampling.reads
    assert "sampling.quadrature.weights_nm" in sampling.reads
    assert "scene.star" in sampling.reads
    assert not any(r.startswith("scene.dynamics") for r in sampling.reads)

    mine = ("sampling.quadrature.nodes_nm",)
    by_hand = SpectralSampling(
        sampling.quadrature,
        sampling.resolution,
        reads=mine,
        built_at=sampling.built_at,
    )
    assert by_hand.reads == mine


def test_sampling_builds_without_a_scene():
    """Fails if a sampling needs a scene, or writes scene stamps or reads without one.

    A scene-less sampling holds no resolution, declares no reads and carries no
    scene layout stamp, and its nodes equal those of the same event built with a
    smooth scene.
    """
    instrument = _instrument()
    smooth = make_system(smooth_planet=True)
    event = cp.build_exposure(
        _BANDS, epoch=smooth.epoch, instrument="imager", time_d=0.0
    )
    bare = cp.build_sampling(instrument, event)
    assert not bare.resolution.keys
    assert bare.reads == ()
    assert "scene_layout" not in dict(bare.built_at)
    with_scene = cp.build_sampling(instrument, event, scene=smooth)
    np.testing.assert_array_equal(
        bare.quadrature.nodes_nm, with_scene.quadrature.nodes_nm
    )
