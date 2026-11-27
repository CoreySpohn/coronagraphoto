"""Mixed-precision regressions for image and aperture rendering."""

from dataclasses import replace
from inspect import unwrap

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import exposure_for, make_path, make_system, with_zodi, worked_setup
from hwoutils.leaves import canonical
from hwoutils.precision import PrecisionError
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff import Detector, FramePlan, IdealDetector, Pointing
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    MultiBandCoronagraph,
    ProductGrids,
    SourcePSFs,
)
from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.spectral import Band, Quadrature, Resolution
from orbix.system import SkyFrame
from packaging.version import Version
from skyscapes import AbstractLocalZodi, PrecomputedZodi, Scene
from skyscapes._sample_time import SampleOrigin, with_sample_origin
from skyscapes.disk import AYOExozodi
from skyscapes.scene import FlatStar, Sources

import coronagraphoto as cp
from coronagraphoto import simulation
from coronagraphoto.aperture import _bilinear, _linear, _on_detector, _ring_moments


class _TentCoronagraph(AbstractImagingCoronagraph):
    """Manufactured separable tent PSFs and uniform stellar and sky maps."""

    grid: PixelGrid = eqx.field(static=True)
    bandpass: Band | None = eqx.field(static=True, default=None)
    band_averaged: bool = eqx.field(static=True, default=True)

    @property
    def frame(self):
        return self.grid.frame

    def at_wavelength(self, wavelength_nm):
        return ProductGrids(self.grid, self.grid, self.grid)

    def stellar_map(self, wavelength_nm, stellar_diameter_lod, *, grid=None):
        grid = self.grid if grid is None else grid
        leakage = 1e-8 * (1 + stellar_diameter_lod**2)
        return Image(
            jnp.broadcast_to(leakage, grid.shape),
            grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        grid = self.grid if grid is None else grid
        row, column = jnp.indices(grid.shape, dtype=x_lod.dtype)
        x, y = grid.pixel_to_unit(column, row)
        tx = jnp.maximum(1 - jnp.abs(x - x_lod[:, None, None]) / 2, 0)
        ty = jnp.maximum(1 - jnp.abs(y - y_lod[:, None, None]) / 2, 0)
        valid = (jnp.abs(x_lod) <= 2.5) & (jnp.abs(y_lod) <= 2.5)
        image = Image(
            jnp.where(valid[:, None, None], 0.01 * tx * ty, 0),
            grid,
            ("source",),
            Measure.integral(),
            "stellar_flux_fraction",
        )
        return SourcePSFs(image, valid)

    def uniform_sky_map(self, wavelength_nm, *, grid):
        dtype = jnp.result_type(wavelength_nm, *jax.tree.leaves(grid.basis))
        data = jnp.full(grid.shape, 0.25, dtype=dtype)
        return Image(data, grid, (), Measure.intensive(), "transmission")


class _UniformZodi(AbstractLocalZodi):
    """Knotless sky with a linear time dependence for aperture reads."""

    flux_scale: jax.Array

    def spectral_breaks_nm(self):
        return np.empty(0)

    def _photon_radiance(self, wavelength_nm, sky_frame, time_d):
        time = jnp.asarray(time_d)
        return jnp.broadcast_to(1e12 * (1 + time / 8), wavelength_nm.shape)


@pytest.fixture(scope="module")
def render_problem():
    """A two-node band at nonzero roll with one planet and a uniform local zodi."""
    wave = jnp.array([500.0, 600.0])
    scene = with_zodi(
        make_system(
            star=FlatStar(1e6, 0.01),
            contrast=0.025,
            smooth_planet=True,
            separation_arcsec=0.0125,
            sky_pa_deg=20.0,
        ),
        PrecomputedZodi(wave, jnp.full(wave.shape, 1e12)),
    )
    instrument = make_path(n=12, step=0.5, detector_n=6)
    instrument = replace(
        instrument, coronagraph=_TentCoronagraph(instrument.coronagraph.grid)
    )
    exposure = cp.build_exposure(
        scene,
        instrument,
        (Band(545, 555),),
        time_d=0.0,
        roll_deg=17.0,
        plan=FramePlan(10.0, 2),
        order=2,
        field_lod=0.0,
    )
    response = cp.aperture_response(
        instrument,
        exposure.quadrature,
        n_separations=3,
        n_angles=2,
        max_separation_arcsec=0.025,
    )
    return canonical((scene, instrument, exposure, response), precision="float64")


@pytest.fixture
def closed_render_problem(render_problem, monkeypatch):
    """Supply closed collaborators at the pinned upstream precision boundaries."""
    scene, instrument, exposure, response = render_problem
    scene = with_zodi(scene, _UniformZodi(jnp.ones(())))
    detector = Detector(
        pixel_scale_arcsec=instrument.detector.pixel_scale_arcsec,
        shape=instrument.detector.shape,
        qe=0.8,
        survival_fraction=0.9,
    )
    offsets = np.asarray(scene.positions(jnp.atleast_1d(exposure.time_d)))

    def fixed_positions(self, time_d, *, ephemeris=None):
        dtype = jnp.result_type(self.dist_pc, time_d)
        return jnp.broadcast_to(
            jnp.asarray(offsets, dtype=dtype), (2, self.n_planets, time_d.size)
        )

    def smooth_sources(self, quadrature, resolution, time_d, *, ephemeris=None):
        star = self.star
        flux = star.node_flux_density(
            quadrature, resolution, time_d, dist_pc=self.dist_pc
        )
        planets = jnp.broadcast_to(
            0.025 * flux[:, :, None, :],
            (*quadrature.nodes_nm.shape, self.n_planets, time_d.size),
        )
        sky = self.local_zodi
        return Sources(
            unscaled_star=flux,
            star_scale=jnp.ones_like(star.flux_scale),
            planets=planets,
            local_zodi=None
            if sky is None
            else jax.vmap(
                lambda t: sky.node_radiance(quadrature, resolution, self.sky_frame, t),
                out_axes=-1,
            )(time_d),
            star_diameter_arcsec=star.angular_diameter_arcsec(self.dist_pc),
            star_luminosity_Lsun=None,
            calibration=jnp.empty((0,), dtype=flux.dtype),
            calibration_bands=(),
        )

    def interior_coverage(source_grid, target_grid, **kwargs):
        # Every detector pixel is interior to the 12x12 native product.
        return jnp.ones(target_grid.shape, dtype=bool)

    monkeypatch.setattr(Scene, "ephemeris", lambda self, time_d: None)
    monkeypatch.setattr(Scene, "positions", fixed_positions)
    monkeypatch.setattr(Scene, "sources", smooth_sources)
    monkeypatch.setattr(simulation, "coverage", interior_coverage)
    return scene, replace(instrument, detector=detector), exposure, response


def _observe(key, scene, instrument, exposure):
    """Compare the electron array; grid metadata belongs to the concrete roll."""
    return cp.observe(key, scene, instrument, exposure).data


def _observation_pair(keys, scene, instrument, exposure, other_scene, other_exposure):
    """Read the target and another roll or reference-star exposure."""
    return (
        _observe(keys[0], scene, instrument, exposure),
        _observe(keys[1], other_scene, instrument, other_exposure),
    )


_DRAWS = ("observe", "roll_pair", "reference_star")


def _render_case(entry, problem):
    """Choose the public read and data for one render entry."""
    scene, instrument, exposure, response = problem
    if entry in {"photon_rates", "expected"}:
        read = getattr(cp, entry)
        args = (scene, instrument, exposure)
    elif entry == "multiband_photon_rates":
        grid = instrument.coronagraph.grid
        children = (
            _TentCoronagraph(grid, bandpass=band, band_averaged=False)
            for band in (Band(500, 550), Band(550, 600))
        )
        instrument = replace(instrument, coronagraph=MultiBandCoronagraph(children))
        read = cp.photon_rates
        args = (scene, instrument, exposure)
    elif entry == "aperture_record":
        read = cp.aperture_record
        args = (scene, instrument, exposure, response)
    elif entry == "scene_aperture_rates":
        read = cp.scene_aperture_rates
        record = cp.aperture_record(scene, instrument, exposure, response)
        args = (record, scene, jnp.array([0.0, 0.5]))
    elif entry == "observe":
        read = _observe
        args = (jax.random.key(0), scene, instrument, exposure)
    else:
        read = _observation_pair
        if entry == "roll_pair":
            other = scene
            other_exposure = replace(exposure, pointing=Pointing(39.0))
        else:
            other = canonical(
                make_system(
                    star=FlatStar(2e6, 0.01),
                    planet_groups=(),
                    separation_arcsec=jnp.empty(0),
                ),
                precision="float64",
            )
            other_exposure = exposure
        args = (
            jax.random.split(jax.random.key(0)),
            scene,
            instrument,
            exposure,
            other,
            other_exposure,
        )
    return read, args


def _mean_draw(self, key, rates, plan):
    """Use the detector's deterministic count mean instead of random samples."""
    return self.mean_and_variance(rates, plan)[0]


@pytest.mark.float64_only(reason="float32 traces and float64 reference in one process")
@pytest.mark.medium(reason="paired scene and detector compilation across entry points")
@pytest.mark.parametrize(
    "entry",
    [
        "photon_rates",
        "multiband_photon_rates",
        "expected",
        *_DRAWS,
        "aperture_record",
        "scene_aperture_rates",
    ],
)
def test_render_precision_closed(render_problem, entry, monkeypatch, request):
    """Compare nontrivial fields and positive-frame means at their peak scale.

    The tent is 1-Lipschitz in each axis before its factor of 1/2. Allow
    32 roundings for the orbit and angular/grid transforms, 16 for the tent
    and radiometric products, 8 for QE and the two-node sum, and up to 49 tap
    products and sums for a 7x7 aperture. 192 roundings cover their composition,
    the aperture interpolations and the observation plan's count conversion.
    The zero-noise detector mean is the total electron rate times live time.

    Fails if a renderer widens float32 inputs or drops an observation frame.
    """
    if entry in _DRAWS:
        monkeypatch.setattr(Detector, "draw", _mean_draw)
    read, args = _render_case(entry, render_problem)
    read = eqx.filter_jit(read)
    precision_pair(read, args, bound=float(ulps(192, np.float32(1))))
    if entry in _DRAWS:
        _, scene, instrument, exposure, *other = args
        frames = (read(*args),) if not other else read(*args)
        targets = (
            ((scene, exposure), (other[0], other[1])) if other else ((scene, exposure),)
        )
        for frame, (target, view) in zip(frames, targets, strict=True):
            assert int(view.plan.n_frames) > 0
            expected = (
                cp.expected(target, instrument, view).total() * view.plan.live_time_s
            )
            assert np.any(np.asarray(expected) > 0)
            np.testing.assert_allclose(
                frame, expected, rtol=0, atol=float(ulps(1, expected.max()))
            )
        if other:
            assert not np.array_equal(*frames)

    read, args = _render_case(entry, request.getfixturevalue("closed_render_problem"))
    assert_precision_closed(eqx.filter_jit(read), *canonical(args, precision="float32"))


@pytest.mark.float64_only(reason="wide scene and narrow render in one process")
@pytest.mark.parametrize(
    "entry",
    ["expected", "photon_rates", "multiband_photon_rates", "disk", "aperture"],
)
def test_wide_scene_narrow_render(render_problem, entry):
    """Scene rounding may move a tent sample by a few ulps of its peak.

    Allow 32 float32 epsilons of each component's peak for orbit evaluation,
    the sky rotation, interpolation and the disk or aperture sum.
    """
    scene, instrument, exposure, response = render_problem
    if entry == "disk":
        scene = make_system(
            disk=AYOExozodi(level_zodis=3.0),
            smooth_planet=True,
            separation_arcsec=0.0125,
        )
        instrument = make_path(n=8, step=0.25, detector_n=4, cube=True)
        exposure = cp.build_exposure(
            scene, instrument, exposure.quadrature.bands, time_d=0.0, order=2
        )
    instrument, exposure, response = canonical(
        (instrument, exposure, response), precision="float32"
    )

    def render(scene):
        if entry == "aperture":
            record = cp.aperture_record(scene, instrument, exposure, response)
            assert record.star.dtype == np.float32
            return cp.scene_aperture_rates(record, scene, np.array([0.0, 0.5]))
        read, args = _render_case(
            "photon_rates" if entry == "disk" else entry,
            (scene, instrument, exposure, response),
        )
        return read(*args)

    mixed = render(scene)
    narrow = render(canonical(scene, precision="float32"))
    for actual, expected in zip(
        jax.tree.leaves(mixed), jax.tree.leaves(narrow), strict=True
    ):
        if jnp.issubdtype(actual.dtype, jnp.floating):
            assert actual.dtype == expected.dtype == np.float32
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=0,
                atol=float(ulps(32, jnp.max(jnp.abs(expected)))),
            )
        else:
            np.testing.assert_array_equal(actual, expected)
    if entry == "disk":
        assert mixed.exozodi.max() > 0


@pytest.mark.float64_only(reason="trace the instrument side of a wide scene read")
def test_wide_scene_node_precision_closed(closed_render_problem, monkeypatch):
    """Trace the node kernel with the positions and spectra from a wide scene.

    The scene and the width conversions themselves contain float64 arithmetic.
    The returned kernel, including its pixel geometry, must be float32 throughout.
    """
    scene, instrument, exposure, _ = closed_render_problem
    instrument, exposure = canonical((instrument, exposure), precision="float32")
    node_function = simulation._node_function
    checked = []

    def check_node(*args):
        node = node_function(*args)
        wavelength = exposure.quadrature.nodes_nm[0, 0]
        assert_precision_closed(node, wavelength, 0, 0)
        checked.append(True)
        return node

    monkeypatch.setattr(simulation, "_node_function", check_node)
    rates = cp.photon_rates(scene, instrument, exposure)
    assert checked
    assert rates.planet.dtype == np.float32


@pytest.mark.float64_only(reason="mixed owner requires x64")
@pytest.mark.parametrize("owner", ["instrument", "exposure"])
def test_render_refuses_mixed_owner(render_problem, owner):
    """A single off-precision leaf in either owner names the repair at entry."""
    scene, instrument, exposure, _ = render_problem
    if owner == "instrument":
        instrument = eqx.tree_at(
            lambda i: i.flux_scale,
            instrument,
            instrument.flux_scale.astype(jnp.float32),
        )
    else:
        exposure = eqx.tree_at(
            lambda e: e.pointing.roll_deg,
            exposure,
            exposure.pointing.roll_deg.astype(jnp.float32),
        )
    with pytest.raises(PrecisionError, match="canonical"):
        cp.expected(scene, instrument, exposure)


@vv.case(
    "image-radiometry",
    "code-verification",
    basis="spike",
    srq="total electron-rate image and detector-summed planet electron rate",
)
@pytest.mark.float64_only(reason="precision_pair needs x64")
@pytest.mark.medium(reason="render the 1600-node worked spectrum at both precisions")
@pytest.mark.parametrize(("quantity", "bound"), [("image", 4.4e-6), ("planet", 1.6e-6)])
def test_worked_expected_precision(quantity, bound):
    """Measure the float32 error for the AB=5 star and planet at 0.05 arcsec.

    At zero roll, the 48x48 detector integrates 1600 nodes over 500-600 nm.
    The largest pixel difference was 1.46e-6 of the float64 image peak.
    The summed planet rate differed by 5.11e-7 of its float64 value.
    Three times each measurement, rounded up to two significant figures,
    gives bounds of 4.4e-6 and 1.6e-6 respectively.

    Fails if float32 rendering loses pixel or planet photometric precision.
    """
    scene, instrument = worked_setup()
    quadrature = Quadrature.build((Band(500, 600),), scene, instrument)
    args = canonical(
        (scene, instrument, exposure_for(scene, quadrature)), precision="float64"
    )

    def render(scene, instrument, exposure):
        dtype = exposure.quadrature.weights_nm.dtype
        rates = cp.expected(scene, instrument, exposure)
        value = rates.total() if quantity == "image" else rates.planet.sum()
        assert value.dtype == dtype
        assert value.max() > 0
        return value

    # The default scale is the float64 peak of the single output leaf.
    precision_pair(render, args, bound=bound, scale=None if quantity == "image" else 0)


class _PoissonWidening(AssertionError):
    """The actual detector sampler call widened inside JAX."""


@pytest.mark.float64_only(reason="float32 sampler traced with x64 available")
@pytest.mark.parametrize("entry", _DRAWS)
@pytest.mark.xfail(
    Version(jax.__version__) < Version("0.12.1"),
    strict=True,
    raises=_PoissonWidening,
    reason="JAX exact Poisson widens its uniform bounds",
)
def test_observe_exact_sampler_precision(closed_render_problem, entry, monkeypatch):
    """Trace the real exact sampler through each image read.

    Fails if the caller widens before or after the sampler, or if the exact
    sampler widens on JAX 0.12.1 or later.
    """
    scene, instrument, exposure, response = closed_render_problem
    assert int(exposure.plan.n_frames) > 0
    ideal = IdealDetector(
        pixel_scale_arcsec=instrument.detector.pixel_scale_arcsec,
        shape=instrument.detector.shape,
    )
    monkeypatch.setattr(
        IdealDetector,
        "quantum_efficiency",
        lambda self, wavelength: jnp.ones(
            jnp.shape(wavelength), self.survival_fraction.dtype
        ),
    )
    poisson = jax.random.poisson
    code = unwrap(poisson).__code__
    # The closure helper falls back to the function definition for an
    # unattributed nested literal. Here that function is the sampler itself.
    location = f"{code.co_filename}:{code.co_firstlineno}"
    matched = f"{location}: precision widened to float64"
    calls = []
    widenings = []

    def check_sampler(key, mean):
        calls.append(mean.dtype)
        assert mean.dtype == np.float32
        inputs = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (key, mean))
        try:
            assert_precision_closed(poisson, *inputs)
        except AssertionError as error:
            if str(error) != matched or not code.co_filename.endswith(
                "/jax/_src/random/core.py"
            ):
                raise
            widenings.append(str(error))
            # Finish the caller trace so a second, local promotion still fails.
            return jnp.zeros_like(mean, dtype=jnp.result_type(int))
        return poisson(key, mean)

    monkeypatch.setattr(jax.random, "poisson", check_sampler)
    # Give each parametrized trace a fresh draw stage so the patched sampler
    # runs even when the previous trace had identical input shapes.
    native_draw = unwrap(simulation._draw)

    @eqx.filter_jit
    def fresh_draw(detector, key, electrons, plan):
        return native_draw(detector, key, electrons, plan)

    monkeypatch.setattr(simulation, "_draw", fresh_draw)
    read, args = _render_case(
        entry, (scene, replace(instrument, detector=ideal), exposure, response)
    )
    assert_precision_closed(eqx.filter_jit(read), *canonical(args, precision="float32"))
    assert calls
    if widenings:
        raise _PoissonWidening(widenings[0])


@pytest.mark.float64_only(reason="float32 traces and float64 reference in one process")
def test_aperture_response_precision_closed(render_problem):
    """The host build casts its tables and its ring reads use the same dtype.

    Host circle overlaps and ring reductions retain binary64. Allow 48
    roundings for narrowed geometry, two tent evaluations and the weighted
    sum over 36 detector pixels. Scale small differences by the peak
    throughput since subtracting nearly equal ring samples can make the
    standard deviation arbitrarily small. Coordinates and weights enter
    as host values, as in the response builder.

    Fails if any response table is stored wide or the ring kernel widens.
    """
    _, instrument, exposure, response = render_problem
    quadrature = exposure.quadrature
    args = (instrument, quadrature)

    def build(instrument, quadrature):
        result = cp.aperture_response(
            instrument,
            quadrature,
            n_separations=3,
            n_angles=2,
            max_separation_arcsec=0.025,
        )
        for name in (
            "radius_pix",
            "separation_step_arcsec",
            "throughput",
            "throughput_ring_std",
            "kernels",
            "probe",
        ):
            assert getattr(result, name).dtype == quadrature.weights_nm.dtype, name
        assert result.throughput_supported.dtype == np.bool_
        assert result._separation_limit_words.dtype == np.uint32
        return result

    bound = float(ulps(48, np.float32(1)))
    precision_pair(build, args, bound=bound, scale=float(response.throughput[0].max()))
    instrument32, quadrature32 = canonical(args, precision="float32")

    def read_ring(model, detector, wavelengths, center):
        return _ring_moments(
            model,
            detector,
            wavelengths,
            center,
            np.array([0.0125]),
            np.array([0.0]),
            np.ones((1, 6, 6)),
        )

    assert_precision_closed(
        read_ring,
        instrument32.coronagraph,
        instrument32.detector,
        quadrature32.nodes_nm[0],
        np.float32(550),
    )


@pytest.mark.float64_only(reason="float32 and float64 host reads in one process")
def test_planet_source_support_precision(render_problem):
    """Planets on either side of the source domain are classified exactly.

    Fails if a nonempty planet axis is dropped or the support kernel widens.
    """
    scene, instrument, exposure, _ = render_problem
    distant = make_system(separation_arcsec=0.065, smooth_planet=True)
    for system, supported in ((scene, True), (distant, False)):
        args = (system, instrument, exposure)
        precision_pair(cp.planet_source_support, args, bound=0)
        result = cp.planet_source_support(*args)
        np.testing.assert_array_equal(result, np.full((1, 1, 1), supported))
        system32, instrument32, exposure32 = canonical(args, precision="float32")
        # Carry roll as a dynamic input rather than closing over the host grid.
        positions = system32.positions(jnp.atleast_1d(exposure32.time_d))

        def support(model, roll, center, positions):
            return simulation._planet_support(model, Pointing(roll), center, positions)

        support_args = canonical(
            (
                instrument32.coronagraph,
                exposure32.pointing.roll_deg,
                jnp.asarray(550.0),
                positions,
            ),
            precision="float32",
        )
        assert_precision_closed(support, *support_args)
        np.testing.assert_array_equal(support(*support_args), [[supported]])


@pytest.mark.float64_only(reason="integer nodes with float32 model under x64")
def test_integer_node_precision(closed_render_problem):
    """Integer nodes leave a half-integer band center and nonzero sky read.

    Fails if nodes, weights or the owner are ignored when choosing precision.
    """
    scene, instrument, exposure, _ = closed_render_problem
    bands = (Band(549, 552),)
    nodes = jnp.array([[550, 551]], dtype=jnp.int32)
    weights = jnp.array([[1.5, 1.5]], dtype=jnp.float32)
    exposure = exposure_for(
        scene, Quadrature(bands, nodes, weights), roll_deg=exposure.pointing.roll_deg
    )
    scene32, instrument32, exposure32 = canonical(
        (scene, instrument, exposure), precision="float32"
    )
    assert_precision_closed(cp.photon_rates, scene32, instrument32, exposure32)
    wide = canonical((scene, instrument, exposure), precision="float64")
    precision_pair(cp.photon_rates, wide, bound=float(ulps(192, np.float32(1))))
    assert np.any(np.asarray(cp.photon_rates(*wide).local_zodi) > 0)
    for dtype in (np.float32, np.float64):
        rule = Quadrature(bands, nodes, weights.astype(dtype))
        assert simulation._band_center_dtype(rule, instrument32) == dtype
        assert simulation._band_center_dtype(rule, instrument) == np.float64
    # With floating nodes, weights still contribute to the working precision.
    rule = Quadrature(bands, nodes.astype(np.float32), weights.astype(np.float64))
    assert simulation._band_center_dtype(rule, instrument32) == np.float64


@pytest.mark.float64_only(reason="check the float32 edge decision with x64 available")
def test_aperture_lattice_edge_precision_closed():
    """The edge and its binary64 neighbor share one float32 admission bin.

    Fails if two widths disagree at a detector edge in eager or compiled reads.
    """
    radius = np.array(
        [1.5, np.nextafter(1.5, 2.0), np.nextafter(np.float32(1.5), np.float32(2.0))]
    )

    def inside(radius):
        return _on_detector(radius, 6, 6)[:, 8, 8]

    args = (jnp.asarray(radius, dtype=jnp.result_type(float)),)
    narrow = canonical(args, precision="float32")
    assert_precision_closed(inside, *narrow)
    for read in (inside, jax.jit(inside)):
        precision_pair(read, args, bound=0)
        np.testing.assert_array_equal(read(*args), [True, True, False])


@pytest.mark.float64_only(reason="float32 interpolation traces inside x64")
def test_aperture_interpolation_precision_closed():
    """Linear and bilinear reads of an affine table are exact at binary offsets.

    Fails if indexing or interpolation widens narrow queries under x64.
    """
    table = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    row, column = jnp.float32(0.25), jnp.float32(1.5)
    assert_precision_closed(_linear, table[0], column)
    assert_precision_closed(_bilinear, table, row, column)
    np.testing.assert_array_equal(_linear(table[0], column), (1.5, True))
    np.testing.assert_array_equal(_bilinear(table, row, column), (2.25, True))


@pytest.mark.float64_only(
    reason="cold float32 render and float64 default in one process"
)
def test_cold_thermal_precision(render_problem):
    """Absent thermal light has the same dtype as present stellar light.

    Fails if the native cold instrument creates a default-width thermal plane.
    """
    scene, instrument, exposure, _ = render_problem
    assert instrument.warm_emission is None
    args = canonical((scene, instrument, exposure), precision="float32")
    rates = cp.photon_rates(*args)
    assert rates.thermal.dtype == np.float32
    np.testing.assert_array_equal(rates.thermal, 0)


def _aperture_problem():
    """A two-ring record with nonzero aperture and background gradients."""
    quadrature = Quadrature((Band(550, 550),), [[550.0]], [[1.0]])
    ring_response = jnp.array([0.5, 1.0]).reshape(1, 1, 1, 2)
    lattice = jnp.linspace(0.5, 1.0, 16).reshape(1, 1, 1, 4, 4)
    return cp.ApertureRecord(
        quadrature=quadrature,
        pointing=Pointing(0.0),
        pixel_scale_arcsec=jnp.asarray(0.125),
        separation_step_arcsec=jnp.asarray(0.125),
        planet_response=jnp.concatenate((ring_response, ring_response / 2)),
        star=jnp.array([[8.0]]),
        star_leakage=jnp.concatenate((lattice, lattice / 2)),
        uniform_sky=jnp.concatenate((lattice / 8, lattice / 16)),
        exozodi=jnp.concatenate((lattice / 4, lattice / 8)),
        n_pix=jnp.concatenate((lattice[:, :, 0], lattice[:, :, 0] / 2)),
        unsupported=jnp.zeros((1, 4, 4)),
        planet_unsupported=jnp.zeros((1, 1, 2)),
        thermal=jnp.array([[0.25]]),
        local_zodi=None,
        sky_frame=SkyFrame(10.0),
        resolution=Resolution((), (), ()),
        reads=(),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_sampled_aperture_host_dates(precision, compiled):
    """Binary64 host dates that alias in float32 give distinct aperture reads.

    Fails if dates are rounded to their shared float32 value before subtraction.
    """
    origin = 1048576.0
    dates = [origin, origin + 0.03125]
    assert np.float32(dates[0]) == np.float32(dates[1])
    record = _aperture_problem()
    scene = make_system(separation_arcsec=0.03125, smooth_planet=True, sky_pa_deg=20.0)
    sampled = with_sample_origin(scene, SampleOrigin(origin))
    # The same orbit with no sample origin reads the exactly known offsets.
    record, scene, sampled = canonical((record, scene, sampled), precision="float32")
    read = (
        eqx.filter_jit(cp.scene_aperture_rates) if compiled else cp.scene_aperture_rates
    )
    expected = read(record, scene, [0.0, 0.03125]).star_leakage
    assert expected[0, 0, 0, 0] != expected[0, 0, 0, 1]
    for host_dates in (dates, tuple(dates)):
        actual = read(record, sampled, host_dates).star_leakage
        np.testing.assert_array_equal(actual, expected)
    if not compiled or precision == np.dtype("float64"):
        # A concrete array is host data eagerly and a binary64 input to JIT.
        actual = read(record, sampled, np.asarray(dates)).star_leakage
        np.testing.assert_array_equal(actual, expected)
    if precision == np.dtype("float64"):
        actual = read(record, sampled, jnp.asarray(dates)).star_leakage
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.float64_only(reason="float32 traces and float64 reference in one process")
@pytest.mark.parametrize(
    "position_input", ["offsets", "prepared", "host-contrast", "host-time"]
)
def test_aperture_rates_precision_closed(position_input):
    """Allow 32 roundings for the two interpolations and rate products.

    The binary-exact offsets, pitch and zero roll make the lattice conversion
    exact. The two bilinear stages each use two differences, a product and a
    sum. The ring read adds a norm, a normalized offset and a linear blend.
    The remaining budget covers the flux, variance and thermal products.

    Fails if interpolation, Python contrast, or host time widens a read.
    """
    record = _aperture_problem()
    east, north, contrast = map(jnp.asarray, (0.03125, 0.0, 0.125))
    if position_input == "prepared":
        position = record.prepare_position(east, north)
        precision_pair(
            lambda r, e, n: r.prepare_position(e, n), (record, east, north), bound=0
        )

        def read(record, position, contrast):
            return cp.aperture_rates(record, position, contrast=contrast, time_d=0.0)

        args = (record, position, contrast)
    elif position_input == "host-contrast":

        def read(record, east, north):
            return cp.aperture_rates(record, east, north, contrast=[[0.125]], time_d=0)

        args = (record, east, north)
    elif position_input == "host-time":
        record = replace(record, local_zodi=_UniformZodi(jnp.ones(())))

        def read(record, east, north, contrast):
            return cp.aperture_rates(record, east, north, contrast=contrast, time_d=0.5)

        args = (record, east, north, contrast)
    else:

        def read(record, east, north, contrast):
            return cp.aperture_rates(record, east, north, contrast=contrast, time_d=0.0)

        args = (record, east, north, contrast)

    read = eqx.filter_jit(read)
    narrow = canonical(args, precision="float32")
    assert_precision_closed(read, *narrow)
    precision_pair(read, args, bound=float(ulps(32, np.float32(1))))
    # The offsets and contrast are exactly representable in both widths.
    mixed = read(narrow[0], *args[1:])
    for actual, expected in zip(
        jax.tree.leaves(mixed), jax.tree.leaves(read(*narrow)), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
