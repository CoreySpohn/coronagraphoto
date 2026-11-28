"""Regression guards for the traced renderer: arguments, compiles and gradients."""

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    GaussianCoronagraph,
    make_path,
    make_system,
    observatory_for,
    with_instrument,
)
from hwoutils.epoch import Epoch
from hwoutils.precision import PrecisionError
from hwoutils.provenance import StampMismatchError
from optixstuff import (
    Exposure,
    FramePlan,
    IdealDetector,
    MultiBandCoronagraph,
    Pointing,
    SimplePupil,
    SourcePSFs,
    SpectralSampling,
    Telescope,
    TransmissiveElement,
)
from optixstuff.grids import Frame
from optixstuff.spectral import Band, Quadrature, Transmission
from skyscapes.scene import FlatStar

import coronagraphoto
from coronagraphoto import (
    aperture_record,
    aperture_response,
    build_exposure,
    build_sampling,
    expected,
    observe,
    photon_rates,
    planet_source_support,
)

_EPOCH = Epoch(2451545.0)


def _event(bands, *, time_d=0.0, roll_deg=0.0, epoch=_EPOCH, plan=None):
    """A commanded event on the one configuration of the test observatories."""
    return Exposure(
        time_d,
        Pointing(roll_deg),
        plan,
        epoch=epoch,
        instrument="imager",
        bands=bands,
    )


@pytest.mark.parametrize("reader", ["expected", "observe", "aperture_record"])
@pytest.mark.parametrize("compile_reader", [False, True], ids=["eager", "jit"])
def test_render_refuses_explicit_pointing_offset(reader, compile_reader):
    """Even a zero offset array is refused when the exposure is traced.

    Building the event with a zero offset constructs; the refusal fires when
    it is rendered. An event with no offset renders.
    """
    scene = make_system(star=FlatStar(jnp.asarray(1e6)))
    path = make_path(n=8, step=0.5)
    exposure = build_exposure(
        (Band(540.0, 560.0),),
        epoch=scene.epoch,
        instrument="imager",
        time_d=0.0,
        roll_deg=17.0,
        offset_mas=(0.0, 0.0),
        plan=FramePlan(10.0, 1.0),
    )
    sampling = build_sampling(path, exposure, field_lod=0.0, scene=scene)
    assert exposure.pointing.roll_deg == 17.0
    np.testing.assert_array_equal(exposure.pointing.offset_mas, [0.0, 0.0])
    if reader == "expected":

        def read(e):
            return expected(scene, path, e, sampling=sampling)
    elif reader == "observe":
        key = jax.random.key(0)

        def read(e):
            return observe(key, scene, path, e, sampling=sampling)
    else:
        response = aperture_response(
            path, sampling.quadrature, n_separations=3, n_angles=4
        )

        def read(e):
            return aperture_record(scene, path, e, response)

    run = jax.jit(read) if compile_reader else read
    with pytest.raises(NotImplementedError, match=r"pointing offsets.*offset_mas=None"):
        run(exposure)
    if reader == "expected" and not compile_reader:
        on_axis = build_exposure(
            exposure.bands, epoch=scene.epoch, instrument="imager", time_d=0.0
        )
        assert np.all(
            np.isfinite(expected(scene, path, on_axis, sampling=sampling).planet)
        )


def _renders(caplog):
    """Count compiles of the shared renderer program in the captured log."""
    return sum("Compiling jit(_render)" in r.message for r in caplog.records)


def _with_nodes(quadrature, nodes):
    return eqx.tree_at(lambda q: q.nodes_nm, quadrature, nodes)


@pytest.mark.slow(
    reason="three compiled observation argument forms, about 27 s projected in CI"
)
def test_exposure_argument_matches_closure():
    """Fails if an exposure or detector geometry cannot be a traced argument.

    Observatory, scene and exposure as arguments must match closing over them
    bitwise, and so must a traced roll against a concrete one.
    """
    system = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    path = make_path(n=8, step=0.5)
    path = eqx.tree_at(
        lambda p: (
            p.instruments[0].detector.read_noise_e_rms,
            p.instruments[0].detector.excess_noise_factor_sq,
        ),
        path,
        (jnp.asarray(1.3), jnp.asarray(1.7)),
    )
    bands = (Band(540.0, 560.0),)
    quadrature = Quadrature(
        bands, jnp.array([[543.0, 557.0]]), jnp.array([[7.0, 13.0]])
    )
    sampling = SpectralSampling(quadrature)
    plan = FramePlan(10.0, 1.0)
    key = jax.random.key(0)

    def read(s, p, key, t):
        exposure = _event(bands, time_d=t, roll_deg=17, plan=plan)
        return observe(key, s, p, exposure, sampling=sampling)

    time_d = jnp.asarray(0.0)
    argument = eqx.filter_jit(read)(system, path, key, time_d)
    closure = eqx.filter_jit(lambda s, k, t: read(s, path, k, t))(system, key, time_d)
    exposure = _event(bands, time_d=time_d, roll_deg=17, plan=plan)
    whole = eqx.filter_jit(lambda e: observe(key, system, path, e, sampling=sampling))(
        exposure
    )
    # XLA may fuse the closed-over constants differently; agree to a few ULP.
    for other in (closure, whole):
        for a, b in zip(jax.tree.leaves(argument), jax.tree.leaves(other), strict=True):
            a, b = np.asarray(a), np.asarray(b)
            if np.issubdtype(a.dtype, np.inexact):
                np.testing.assert_array_max_ulp(a, b, maxulp=4)
            else:
                np.testing.assert_array_equal(a, b)
    eager = observe(key, system, path, exposure, sampling=sampling)
    assert eager.exposure.pointing.roll_deg == 17.0
    assert eager.image.grid.orientation is None
    assert whole.image.grid.orientation is None


@pytest.mark.medium(reason="one compile per renderer and one observation sampler")
def test_reader_compiles_once_for_any_values(caplog):
    """Fails on recompilation after time, flux, node, weight or roll changes.

    Each renderer compiles once as its inputs change. The observation compiles
    the electron renderer and both entry points reuse it. Two events that
    differ in epoch and in commanded time, and two samplings that differ in
    stamps and node values, run one program: the static record of the event
    and of the sampling never reaches the compiled stage. A shifted epoch is
    exercised in binary64 only, where the shift is added exactly.
    """
    system = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    varied = eqx.tree_at(lambda s: s.star.flux_phot_per_nm_m2, system, jnp.asarray(2e6))
    path = make_path(n=8, step=0.5, chromatic=True)
    bands = (Band(535.0, 565.0),)
    quadrature = Quadrature(bands, np.array([[550.0]]), np.array([[30.0]]))
    changed = Quadrature(bands, np.array([[551.0]]), np.array([[60.0]]))
    key, plan = jax.random.key(0), FramePlan(1e6, 1.0)
    x64 = bool(jax.config.jax_enable_x64)

    def args(system, rule, time_d, roll, *, epoch=_EPOCH, built_at=()):
        exposure = _event(bands, time_d=time_d, roll_deg=roll, epoch=epoch, plan=plan)
        sampling = SpectralSampling(rule, built_at=built_at)
        return (system, path, exposure), {"sampling": sampling}

    t0, t1 = jnp.asarray(0.0), jnp.asarray(1.0)
    initial = args(system, quadrature, t0, 23)
    later = args(varied, quadrature, t1, 41)
    other_nodes = args(varied, changed, t1, 41, built_at=(("producer", "str:other"),))
    # Another day zero for the same physical instant: the scene epoch is
    # `_EPOCH`, so the event is stated on a later epoch.
    shifted = args(varied, quadrature, t1 - 10.0, 41, epoch=Epoch(2451555.0))
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level("WARNING"):
        for reader in (photon_rates, observe, expected):
            caplog.clear()
            prefix = (key,) if reader is observe else ()
            first = jax.block_until_ready(reader(*prefix, *initial[0], **initial[1]))
            renderer_compiles = 0 if reader is expected else 1
            assert _renders(caplog) == renderer_compiles
            second = jax.block_until_ready(reader(*prefix, *later[0], **later[1]))
            different = jax.block_until_ready(
                reader(*prefix, *other_nodes[0], **other_nodes[1])
            )
            assert _renders(caplog) == renderer_compiles
            if x64:
                again = jax.block_until_ready(
                    reader(*prefix, *shifted[0], **shifted[1])
                )
                assert _renders(caplog) == renderer_compiles
                np.testing.assert_allclose(
                    np.asarray(_field(reader, again)),
                    np.asarray(_field(reader, second)),
                    rtol=1e-12,
                )
            if reader is photon_rates:
                assert first.quadrature is quadrature
            assert np.any(np.asarray(_field(reader, first)) != _field(reader, second))
            assert np.any(
                np.asarray(_field(reader, second)) != _field(reader, different)
            )


def _field(reader, result):
    """The array each reader returns that the test compares."""
    return result.image.data if reader is observe else result.star_leakage


class _CountingCoronagraph(GaussianCoronagraph):
    """A coronagraph that counts how often it is bound to a telescope."""

    calls: ClassVar[list] = []

    def for_telescope(self, telescope):
        """Record the call and bind as the base class does."""
        type(self).calls.append(None)
        return super().for_telescope(telescope)


def test_forty_nodes_and_eight_rolls_compile_one_program(caplog):
    """Fails if a node count or a roll sequence compiles more than one program.

    The coronagraph counts its telescope bindings. Each public call binds on
    the host a fixed number of times, so the count per call is the same for
    the call that traces the program as for the cached ones: a binding made
    per node inside the scan would add to the traced call only.
    """
    system = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    plain = make_path(n=8, step=0.5, chromatic=True)
    path = with_instrument(
        plain,
        coronagraph=_CountingCoronagraph(8, 0.5, chromatic=True),
    )
    scene = system
    band = Band(520.0, 580.0)
    nodes = np.linspace(521.0, 579.0, 40)
    wide = Quadrature((band,), nodes[None], np.full((1, 40), 1.5))
    sampling = SpectralSampling(wide)
    counts = []

    def render(roll):
        before = len(_CountingCoronagraph.calls)
        out = expected(scene, path, _event((band,), roll_deg=roll), sampling=sampling)
        counts.append(len(_CountingCoronagraph.calls) - before)
        return jax.block_until_ready(out)

    with jax.log_compiles(True), caplog.at_level("WARNING"):
        rolls = np.linspace(0.0, 70.0, 8)
        first = render(rolls[0])
        initial_compiles = _renders(caplog)
        assert initial_compiles <= 1
        later = [render(roll) for roll in rolls[1:]]
        assert _renders(caplog) == initial_compiles
    assert not np.allclose(first.planet, later[-1].planet)
    assert len(counts) == 8
    assert len(set(counts)) == 1
    assert counts[0] >= 1


@pytest.mark.medium(reason="four reverse-mode programs and finite differences")
@pytest.mark.float64_only(reason="central finite differences at step 1e-4")
def test_roll_and_node_gradients_match_finite_differences():
    """Fails if a traced roll or node wavelength has a wrong derivative.

    The scalar mixes the planet and the leakage; central differences in
    float64 agree with jax.grad to rtol 1e-5 for the roll and each node, on a
    hand-built sampling whose nodes are the differentiated leaves.
    """
    system = make_system(
        star=FlatStar(jnp.asarray(1e6)), sky_pa_deg=40.0, smooth_planet=True
    )
    path = make_path(n=24, step=0.5, chromatic=True)
    scene = system
    bands = (Band(530.0, 570.0),)
    quadrature = Quadrature(
        bands,
        jnp.array([[537.0, 548.0, 561.0]]),
        jnp.array([[10.0, 20.0, 10.0]]),
    )

    def scalar(roll, nodes):
        exposure = _event(bands, roll_deg=roll)
        sampling = SpectralSampling(_with_nodes(quadrature, nodes))
        rates = expected(scene, path, exposure, sampling=sampling)
        return jnp.sum(rates.planet**2) + 1e-3 * jnp.sum(rates.star_leakage**2)

    roll, nodes = 17.0, quadrature.nodes_nm
    d_roll, d_nodes = jax.grad(scalar, argnums=(0, 1))(roll, nodes)
    h = 1e-4
    fd_roll = (scalar(roll + h, nodes) - scalar(roll - h, nodes)) / (2 * h)
    np.testing.assert_allclose(d_roll, fd_roll, rtol=1e-5)
    for i in range(nodes.shape[1]):
        step = jnp.zeros_like(nodes).at[0, i].set(h)
        fd = (scalar(roll, nodes + step) - scalar(roll, nodes - step)) / (2 * h)
        np.testing.assert_allclose(d_nodes[0, i], fd, rtol=1e-5)


def _two_child_observatory():
    """A two-band coronagraph on a pupil whose diameters match both children."""
    blue = GaussianCoronagraph(n=16, bandpass=Band(490, 510))
    red = GaussianCoronagraph(
        n=20, step=0.35, frame=Frame(600.0, 3.0, "inscribed"), bandpass=Band(590, 610)
    )
    plain = make_path(n=16, step=0.5)
    pupil = SimplePupil(
        circumscribed_diameter_m=6.0, inscribed_diameter_m=3.0, collecting_area_m2=40.0
    )
    (configuration,) = plain.instruments
    return observatory_for(
        pupil,
        configuration.attenuating_elements,
        MultiBandCoronagraph((blue, red)),
        configuration.detector,
    )


def test_traced_nodes_select_the_band_model_inside_the_program():
    """Fails if band selection needs concrete nodes or picks the wrong model.

    Eager calls keep one branch per model the nodes fall in; traced nodes keep
    every model and select from the node wavelength. Both give the same
    images, and a node outside every band gives NaN instead of a wrong model.
    """
    path = _two_child_observatory()
    scene = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    bands = (Band(490.0, 610.0),)
    quadrature = Quadrature(
        bands,
        jnp.array([[495.0, 600.0, 505.0]]),
        jnp.array([[1.0, 1.0, 1.0]]),
    )
    exposure = _event(bands, roll_deg=10.0)

    def render(nodes):
        sampling = SpectralSampling(_with_nodes(quadrature, nodes))
        return photon_rates(scene, path, exposure, sampling=sampling)

    eager = render(quadrature.nodes_nm)
    traced = eqx.filter_jit(render)(quadrature.nodes_nm)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(traced), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    assert not np.allclose(eager.planet[0, 0], eager.planet[0, 1])
    outside = eqx.filter_jit(render)(jnp.array([[550.0] * 3]))
    assert np.all(np.isnan(outside.planet))


def test_two_band_event_routes_each_band_to_its_child():
    """Fails if a band reads the wrong child, or a swapped band order is accepted.

    A two-band event on a two-child coronagraph renders each band bitwise as
    the one-band event renders on the observatory holding that child alone.
    The same bands in the other order, against the sampling built for the
    first order, are refused by the event check before any render.
    """
    path = _two_child_observatory()
    scene = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    bands = (Band(490.0, 510.0), Band(590.0, 610.0))
    both = build_exposure(
        bands, epoch=scene.epoch, instrument="imager", time_d=0.0, roll_deg=10.0
    )
    sampling = build_sampling(path, both, order=2, field_lod=0.0, scene=scene)
    together = expected(scene, path, both, sampling=sampling)
    (configuration,) = path.instruments
    for i, child in enumerate(configuration.coronagraph.coronagraphs):
        alone = with_instrument(path, coronagraph=child)
        one = build_exposure(
            bands[i : i + 1],
            epoch=scene.epoch,
            instrument="imager",
            time_d=0.0,
            roll_deg=10.0,
        )
        rule = build_sampling(alone, one, order=2, field_lod=0.0, scene=scene)
        single = expected(scene, alone, one, sampling=rule)
        np.testing.assert_array_equal(together.planet[i], single.planet[0])
        np.testing.assert_array_equal(together.star_leakage[i], single.star_leakage[0])
    swapped = replace(both, bands=bands[::-1])
    with pytest.raises(ValueError, match="bands differ"):
        expected(scene, path, swapped, sampling=sampling)
    with pytest.raises(ValueError, match="bands differ"):
        planet_source_support(scene, path, swapped, sampling=sampling)


def test_sampling_and_event_must_match_the_hardware():
    """Fails if a configuration name or a build stamp is not checked.

    A name absent from the observatory raises listing the names. A sampling
    stamped for another configuration name raises ``StampMismatchError``. A
    sampling built by hand carries no stamps and is accepted.
    """
    scene = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    path = make_path(n=8, step=0.5)
    bands = (Band(540.0, 560.0),)
    event = build_exposure(bands, epoch=scene.epoch, instrument="imager", time_d=0.0)
    sampling = build_sampling(path, event, order=2, field_lod=0.0, scene=scene)
    absent = replace(event, instrument="ifs")
    with pytest.raises(ValueError, match=r"no instrument named 'ifs'.*imager"):
        expected(scene, path, absent, sampling=sampling)
    two = replace(
        path,
        instruments=(*path.instruments, replace(path.instruments[0], name="ifs")),
    )
    with pytest.raises(StampMismatchError, match="instrument"):
        expected(scene, two, absent, sampling=sampling)
    by_hand = SpectralSampling(sampling.quadrature)
    assert by_hand.built_at == ()
    rates = expected(scene, two, absent, sampling=by_hand)
    assert np.all(np.isfinite(rates.planet))


def test_telescope_break_adds_a_node_to_the_sampling():
    """Fails if the sampling is built from the bare configuration.

    A telescope element with a spectral break inside the band splits the band
    into two pieces, so the sampling holds more nodes than the same
    observatory without that element.
    """
    scene = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    path = make_path(n=8, step=0.5)
    bands = (Band(500.0, 600.0),)
    event = build_exposure(bands, epoch=scene.epoch, instrument="imager", time_d=0.0)
    plain = build_sampling(path, event, order=3, field_lod=0.0, scene=scene)
    coated = replace(
        path,
        telescope=Telescope(
            path.telescope.pupil,
            attenuating_elements=(
                TransmissiveElement(
                    Transmission([450.0, 550.0, 650.0], [0.9, 0.8, 0.9])
                ),
            ),
        ),
    )
    broken = build_sampling(coated, event, order=3, field_lod=0.0, scene=scene)
    assert broken.quadrature.nodes_nm.shape[1] == 2 * plain.quadrature.nodes_nm.shape[1]
    assert len(broken.quadrature.pieces[0]) == 2


def test_observe_returns_exposure_data_with_the_sampling_stamp():
    """Match eager bytes and build facts exactly through one trace for two rolls.

    ``observe`` returns ``ExposureData`` with no variance, every pixel valid, a
    grid without orientation and the event it was given. Two observations
    built from one sampling carry that sampling's ``layout`` stamp exactly, and
    the producer version is stamped. The roll lives on the exposure and one
    program serves two rolls.
    """
    traces = []
    scene = make_system(planet_groups=(), separation_arcsec=jnp.empty(0))
    path = make_path(n=8, step=0.5, detector_n=3)
    detector = IdealDetector(
        pixel_scale_arcsec=path.instruments[0].detector.grid.step, shape=(3, 3)
    )
    path = with_instrument(path, detector=detector)
    exposure = build_exposure(
        (Band(540.0, 560.0),),
        epoch=scene.epoch,
        instrument="imager",
        time_d=0.0,
        plan=FramePlan(10.0, 1.0),
    )
    sampling = build_sampling(path, exposure, order=1, field_lod=0.0, scene=scene)

    @eqx.filter_jit
    def read(key, scene, path, exposure, roll):
        traces.append(None)
        pointed = eqx.tree_at(lambda e: e.pointing.roll_deg, exposure, roll)
        return observe(key, scene, path, pointed, sampling=sampling)

    stamps = []
    for seed, roll_deg in enumerate((17.0, 41.0)):
        key = jax.random.key(seed)
        roll = jnp.asarray(roll_deg, dtype=exposure.pointing.roll_deg.dtype)
        result = read(key, scene, path, exposure, roll)
        pointed = eqx.tree_at(lambda e: e.pointing.roll_deg, exposure, roll)
        eager = observe(key, scene, path, pointed, sampling=sampling)
        actual_values = np.asarray(result.image.data)
        eager_values = np.asarray(eager.image.data)
        assert actual_values.dtype == eager_values.dtype
        assert actual_values.tobytes() == eager_values.tobytes()
        assert result.built_at == eager.built_at
        assert result.image.grid.orientation is None
        assert eager.image.grid.orientation is None
        assert eager.variance is None
        assert bool(np.all(np.asarray(eager.valid)))
        assert eager.flags is None
        assert eager.exposure is pointed
        assert result.exposure.pointing.roll_deg == roll
        stamps.append(result.built_at)

    assert len(traces) == 1
    assert stamps[0] == stamps[1]
    facts = dict(stamps[0])
    assert facts.pop("producer_version").startswith("str:")
    layout = dict(sampling.built_at)["layout"]
    assert layout.startswith("sha256:") and len(layout) == len("sha256:") + 64
    assert facts == {
        "producer": "str:coronagraphoto.observe",
        "detector": "str:IdealDetector",
        "layout": layout,
    }
    assert not hasattr(coronagraphoto, "observe_data")


def test_a_binary32_time_across_epochs_is_refused_and_one_epoch_differentiates(
    precision,
):
    """Fails if the epoch shift follows a binary32 cast, or the clock is dead.

    A binary32 ``time_d`` on an event whose epoch differs from the scene's
    raises ``PrecisionError`` at the render: the shift cannot be formed
    without losing the instant. On the scene's own epoch the same event
    renders, and ``jax.grad`` with respect to ``time_d`` is finite and
    nonzero. In binary64 it matches a central difference at the
    step-optimal width (the cube root of the machine epsilon) to 1e-6
    relative; the planet is on a short circular orbit, so the image moves.
    """
    scene = make_system(
        star=FlatStar(jnp.asarray(1e6)), separation_arcsec=0.02, smooth_planet=True
    )
    path = make_path(n=8, step=0.5)
    bands = (Band(540.0, 560.0),)
    narrow = jnp.asarray(0.5, dtype=jnp.float32)
    shifted = _event(bands, epoch=Epoch(2451555.0))
    shifted = eqx.tree_at(lambda e: e.time_d, shifted, narrow)
    sampling = SpectralSampling(Quadrature.build(bands, path.select("imager"), order=2))
    with pytest.raises(PrecisionError):
        expected(scene, path, shifted, sampling=sampling)

    def planet_sum(time_d):
        event = _event(bands, time_d=time_d, epoch=scene.epoch)
        return jnp.sum(expected(scene, path, event, sampling=sampling).planet)

    time_d = jnp.asarray(0.5, dtype=precision)
    slope = jax.grad(planet_sum)(time_d)
    assert np.isfinite(slope) and slope != 0
    if precision == np.float64:
        h = np.finfo(np.float64).eps ** (1 / 3)
        fd = (planet_sum(time_d + h) - planet_sum(time_d - h)) / (2 * h)
        np.testing.assert_allclose(slope, fd, rtol=1e-6)


class _EastOnlyCoronagraph(GaussianCoronagraph):
    """A source domain that admits only the positive detector x half plane."""

    def _native_source_psfs(self, x_lod, y_lod):
        psfs = super()._native_source_psfs(x_lod, y_lod)
        return SourcePSFs(psfs.image, x_lod > 0)


@pytest.mark.float64_only(reason="a binary32 time cannot be moved between epochs")
def test_planet_source_support_reads_an_event_on_a_shifted_epoch():
    """Fails if the support probe reads the event time without the epoch shift.

    The planet's orbit carries it across the edge of a half plane source domain
    between two dates half a period apart. An event stated on an epoch 10 d
    after the scene's, at the instants that put the planet at the same two
    dates, gives the flags of the event stated on the scene's own epoch, with
    the released ``[band, planet, time]`` shape. The two dates disagree, so a
    probe that ignored the shift would fail. Boolean flags need no tolerance.
    """
    scene = make_system(
        star=FlatStar(jnp.asarray(1e6)), separation_arcsec=0.02, smooth_planet=True
    )
    path = with_instrument(
        make_path(n=8, step=0.5), coronagraph=_EastOnlyCoronagraph(n=8, step=0.5)
    )
    bands = (Band(540.0, 560.0),)
    sampling = SpectralSampling(Quadrature.build(bands, path.select("imager"), order=1))
    dates = jnp.array([0.0, 16.3])
    here = planet_source_support(
        scene, path, _event(bands, time_d=dates), sampling=sampling
    )
    later = planet_source_support(
        scene,
        path,
        _event(bands, time_d=dates - 10.0, epoch=Epoch(2451555.0)),
        sampling=sampling,
    )
    assert here.shape == later.shape == (1, 1, 2)
    assert here[0, 0, 0] != here[0, 0, 1]
    np.testing.assert_array_equal(here, later)
