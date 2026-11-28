"""Aperture response, record and read against closed forms and image sums."""

import copy
import dataclasses
from dataclasses import replace
from fractions import Fraction

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from conftest import (
    GaussianCoronagraph,
    PlateauCoronagraph,
    leinert_radiance,
    make_path,
    make_system,
    observatory_for,
    with_instrument,
    with_zodi,
    write_yip,
)
from hwoutils.constants import Rsun2m, arcsec2rad, c, h, k_B, nm2m
from hwoutils.conversions import photons_per_nm_per_m2_to_jy
from hwoutils.epoch import Epoch
from hwoutils.leaves import canonical
from hwoutils.numerics import eps, roundoff, sum_roundoff
from hwoutils.precision import PrecisionError
from hwoutils.provenance import StampMismatchError, bind_reads, stale_reads
from hwoutils.testing import precision_pair, ulps
from optixstuff import (
    ConstantThroughput,
    Detector,
    FramePlan,
    SimplePupil,
)
from optixstuff.apertures import DEFAULT_APERTURE_RULE, Circular, HalfMaximum
from optixstuff.grids import Frame, Image, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Quadrature
from skyscapes import AYOZodi, LeinertZodi
from skyscapes.physical_model import BlackbodyPhysicalModel, GridPhysicalModel
from skyscapes.scene import BlackbodyStar, FlatStar, Planet, Star
from yippy import open_yip

import coronagraphoto as cp
from coronagraphoto._version import __version__

_RULE = Circular(0.7)
_BANDS = (Band(500.0, 600.0),)


class SlopedLeakage(GaussianCoronagraph):
    """Gaussian planet PSF over a stellar leakage that is linear in x and y."""

    def _native_stellar_map(self, stellar_diameter_lod):
        """Tilt the uniform leakage differently along the two detector axes."""
        flat = super()._native_stellar_map(stellar_diameter_lod)
        x, y = self.grid.coordinates()
        return eqx.tree_at(
            lambda image: image.data, flat, flat.data * (1 + 0.05 * x + 0.02 * y)
        )


# No residual and each background counted once.
_NO_RESIDUAL = ox.PostProcessing(background_variance_factor=1.0)


def _instrument(coronagraph):
    """Worked observatory with 0.3 lambda/D pixels and the given coronagraph."""
    path = make_path(detector_n=48, detector_step=0.3)
    return eqx.tree_at(lambda p: p.instruments[0].coronagraph, path, coronagraph)


def _sel(observatory):
    """The view of the one configuration of a test observatory."""
    return observatory.select("imager")


def _build(scene, observatory, bands, *, time_d=0.0, roll_deg=0.0, **options):
    """The commanded event of a scene and its sampling (options as build_sampling)."""
    exposure = cp.build_exposure(
        bands,
        epoch=scene.epoch,
        instrument="imager",
        time_d=time_d,
        roll_deg=roll_deg,
    )
    sampling = cp.build_sampling(observatory, exposure, scene=scene, **options)
    return exposure, sampling


def _offsets(separation, position_angle_deg):
    """East and north offsets of a planet at a separation and position angle."""
    angle = jnp.deg2rad(position_angle_deg)
    return separation * jnp.sin(angle), separation * jnp.cos(angle)


def _image_weights(observatory, x_arcsec, y_arcsec):
    """The rule's pixel weights for an aperture at a detector position."""
    grid = _sel(observatory).detector.grid
    frame = _sel(observatory).coronagraph.grid.frame
    scale = float(frame.lod_to_arcsec(550.0))
    ones = Image(
        jnp.ones(grid.shape),
        grid.to_lod(550.0, frame),
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    return np.asarray(
        _RULE.weights(ones, (float(x_arcsec) / scale, float(y_arcsec) / scale))
    )


@pytest.fixture(scope="module")
def plateau():
    """Flat-PSF observatory, its response and one target's record."""
    observatory = _instrument(PlateauCoronagraph(40, 0.25))
    scene = make_system(separation_arcsec=0.03)
    # The record tables are compared with the image at the same four nodes, so
    # the layout is not under test here and the field is zero.
    exposure, sampling = _build(scene, observatory, _BANDS, field_lod=0.0)
    response = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        n_separations=5,
        n_angles=4,
        max_separation_arcsec=0.04,
    )
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    assert record.pointing is exposure.pointing
    assert record.sampling is sampling
    return scene, observatory, exposure, sampling, response, record


@pytest.fixture(scope="module")
def sloped():
    """Sloped-leakage observatory with a record rendered at zero roll."""
    observatory = _instrument(SlopedLeakage(40, 0.25))
    scene = make_system(separation_arcsec=0.05)
    # Table and image are compared at the same four nodes; see ``plateau``.
    exposure, sampling = _build(scene, observatory, _BANDS, field_lod=0.0)
    response = cp.aperture_response(
        _sel(observatory), sampling.quadrature, n_separations=9, n_angles=8
    )
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    return scene, observatory, exposure, sampling, response, record


@vv.case("aperture-photon-budget", "code-verification")
def test_worked_example_closed_form(plateau):
    """Fails if a radiometric factor or the fixed-angle aperture area is wrong.

    The manufactured PSF is flat, 0.3/16 per square lambda/D, and the leakage
    is 1e-9 per square lambda/D, so every aperture sum is an area: the circle
    of 0.7 lambda/D at the 550 nm band centre covers pi (0.7 * 550 / lambda)**2
    square lambda/D at node lambda, 19% apart between the outer nodes.
    Basis: exact areas for the rates, within 3e-2 relative: the sharp edges of
    the manufactured fields ring under band-limited resampling onto 0.3
    lambda/D pixels, measured at up to 2.1e-2. The radius allows six
    operations for angular conversion and division. Pixel count allows the
    propagation through the square, plus the aperture-weight reduction depth.
    """
    _, _, _, _, response, record = plateau
    nodes = np.asarray(record.quadrature.nodes_nm)
    area = np.pi * (0.7 * 550.0 / nodes) ** 2
    rates = cp.aperture_rates(record, *_offsets(0.03, 40.0), contrast=1e-10, time_d=0.0)
    incident = 1e6 * 40 * 0.4
    np.testing.assert_allclose(
        response.radius_pix,
        [0.7 / 0.3],
        rtol=roundoff(1, n_ops=6, dtype=response.radius_pix.dtype),
        atol=0,
    )
    np.testing.assert_allclose(
        rates.planet, 1e-10 * incident * 0.3 / 16 * area, rtol=3e-2
    )
    np.testing.assert_allclose(rates.star_leakage, incident * 1e-9 * area, rtol=3e-2)
    n_pix = np.pi * (0.7 / 0.3) ** 2
    np.testing.assert_allclose(
        rates.n_pix,
        [n_pix],
        rtol=0,
        atol=roundoff(n_pix, n_ops=16, dtype=rates.n_pix.dtype)
        + sum_roundoff(n_pix, response.kernels.shape[-1] ** 2, dtype=rates.n_pix.dtype),
    )
    assert 0.5 * n_pix < float(rates.variance.n_pix[0]) < n_pix
    # A flat field under weights w**2 keeps the ratio of the two weight sums.
    np.testing.assert_allclose(
        rates.variance.planet / rates.planet,
        np.broadcast_to(rates.variance.n_pix / rates.n_pix, nodes.shape),
        rtol=3e-2,
    )
    assert rates.carrier == "photon" and rates.spectral
    np.testing.assert_array_equal(rates.thermal, 0.0)
    np.testing.assert_array_equal(rates.stellar_companions, 0.0)
    np.testing.assert_array_equal(rates.local_zodi, 0.0)
    np.testing.assert_array_equal(rates.exozodi, 0.0)


@vv.case("aperture-photon-budget", "code-verification")
def test_background_lattice_matches_image_sums(sloped):
    """Fails if a kernel phase, axis order, roll sign or squared weight is wrong.

    A leakage linear in x and y, with different slopes, makes every aperture
    sum depend on where the aperture sits. Planets are placed on half-pixel
    lattice positions, where the record holds the sum itself, and read at a
    roll the record was not built for. Basis: the same exact pixel weights on
    both sides. The relative bound covers each dot product's reduction
    depth, plus 32 arithmetic steps for rotation, lattice coordinates,
    powers of the weights and the bilinear read.
    """
    scene, observatory, exposure, sampling, _, record = sloped
    photons = cp.photon_rates(scene, observatory, exposure, sampling=sampling)
    pitch = float(_sel(observatory).detector.pixel_scale_arcsec)
    roll = 25.0
    for dx, dy in ((3.0, -2.5), (-4.5, 1.0), (0.5, 6.5)):
        x, y = dx * pitch, dy * pitch
        east, north = SkyOrientation(roll).detector_to_sky(x, y)
        rates = cp.aperture_rates(
            record,
            east,
            north,
            contrast=1e-10,
            time_d=0.0,
            roll_deg=roll,
        )
        weights = _image_weights(observatory, x, y)
        dtype = photons.star_leakage.dtype
        rtol = float(
            roundoff(1, n_ops=32, dtype=dtype)
            + sum_roundoff(2.0, weights.size, dtype=dtype)
        )
        # The aperture has a fractional edge pixel and the intensities differ across
        # it, so the two moments are different numbers.
        edge = (weights > 0) & (weights < 1)
        assert edge.any()
        intensity = np.asarray(photons.star_leakage)[0, 0]
        first = float((intensity * weights).sum())
        second = float((intensity * weights**2).sum())
        assert abs(second / first - 1) > 1e-3
        np.testing.assert_allclose(
            [rates.star_leakage[0, 0], rates.variance.star_leakage[0, 0]],
            [first, second],
            rtol=rtol,
            atol=0,
        )
        for moment, power in ((rates, 1), (rates.variance, 2)):
            np.testing.assert_allclose(
                moment.star_leakage,
                np.einsum(
                    "wqyx,yx->wq", np.asarray(photons.star_leakage), weights**power
                ),
                rtol=rtol,
                atol=0,
            )
            np.testing.assert_allclose(
                moment.n_pix, [float((weights**power).sum())], rtol=rtol, atol=0
            )
    # The slope is seen: two positions differ by far more than the tolerance.
    first, second = (
        cp.aperture_rates(
            record, *_offsets(0.05, pa), contrast=1e-10, time_d=0.0
        ).star_leakage
        for pa in (90.0, 270.0)
    )
    assert np.all(abs(first / second - 1) > 0.05)


@vv.case("aperture-photon-budget", "code-verification")
def test_planet_table_matches_image_sums(sloped):
    """Fails if the planet table loses the star spectrum, gain or a PSF factor.

    The Gaussian PSF does not depend on position, so the ring mean differs
    from one planet only by pixel phase. Basis: the measured ring scatter of
    the response, three standard deviations, plus the linear-interpolation
    error bounded by one eighth of the largest second difference of the
    neighbouring knots. Host float64 weights and accumulation keep the
    image sum independent of the render lane. The scale identities allow
    four operations for the flux, throughput and contrast products.
    """
    scene, observatory, exposure, sampling, response, record = sloped
    photons = cp.photon_rates(scene, observatory, exposure, sampling=sampling)
    east, north = scene.positions(jnp.array([0.0]))[:2, 0, 0]
    separation = float(jnp.hypot(east, north))
    x, y = SkyOrientation(0.0).sky_to_detector(east, north)
    weights = _image_weights(observatory, x, y)
    rates = cp.aperture_rates(
        record,
        east,
        north,
        contrast=1e-10,
        time_d=0.0,
    )
    knot = int(separation / float(response.separation_step_arcsec))
    table = np.asarray(response.throughput[0, 0])
    scatter = np.asarray(response.throughput_ring_std[0])[:, knot : knot + 2].max(-1)
    curvature = abs(np.diff(table[:, knot - 1 : knot + 3], n=2, axis=-1)).max(-1) / 8
    tolerance = (3 * scatter + curvature) / table[:, knot : knot + 2].min(-1)
    truth = np.einsum("wqyx,yx->wq", np.asarray(photons.planet), weights)
    assert np.all(tolerance < 0.05)
    assert np.all(abs((np.asarray(rates.planet) / truth)[0] - 1) <= tolerance)
    # Contrast factors out of the planet and touches nothing else.
    doubled = cp.aperture_rates(
        record, *_offsets(separation, 90.0), contrast=2e-10, time_d=0.0
    )
    single = cp.aperture_rates(
        record, *_offsets(separation, 90.0), contrast=1e-10, time_d=0.0
    )
    np.testing.assert_allclose(
        doubled.planet,
        2 * single.planet,
        rtol=roundoff(1, n_ops=4, dtype=single.planet.dtype),
        atol=0,
    )
    np.testing.assert_array_equal(doubled.star_leakage, single.star_leakage)
    per_node = cp.aperture_rates(
        record,
        *_offsets(separation, 90.0),
        contrast=jnp.full((1, 4), 1e-10),
        time_d=0.0,
    )
    np.testing.assert_allclose(
        per_node.planet,
        single.planet,
        rtol=roundoff(1, n_ops=4, dtype=single.planet.dtype),
        atol=0,
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_read_geometry_on_a_hand_made_record(plateau):
    """Fails if position angle, roll, parity or the sky-fixed exozodi is wrong.

    The tables are set by hand to each lattice position's pixel coordinate,
    which bilinear interpolation reproduces exactly, so every read returns
    where it looked. The expected detector position is written out for
    detector +y at position angle ``roll`` and east to the left at zero roll:
    x = -(E cos r - N sin r), y = E sin r + N cos r. The algebra allows
    24 steps of roundoff for the angle conversions, rotation, pixel offset
    and bilinear interpolation.
    """
    record = plateau[5]
    shape = record.n_pix.shape[-2:]
    rows, columns = jnp.indices(shape) / 2.0
    tables = (
        jnp.broadcast_to(columns, record.star_leakage.shape),
        jnp.broadcast_to(rows, record.uniform_sky.shape),
    )
    record = eqx.tree_at(
        lambda r: (r.star_leakage, r.uniform_sky, r.exozodi),
        record,
        (tables[0], tables[1], tables[0]),
    )
    rtol = roundoff(1, n_ops=24, dtype=record.star_leakage.dtype)
    pitch, centre = float(record.pixel_scale_arcsec), (shape[0] / 2 - 1) / 2

    def expected(separation, angle, roll):
        e = separation * np.sin(np.deg2rad(angle))
        n = separation * np.cos(np.deg2rad(angle))
        c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
        return centre - (e * c - n * s) / pitch, centre + (e * s + n * c) / pitch

    separation, angle = 0.031, 70.0
    built = float(record.pointing.roll_deg)
    rates = cp.aperture_rates(
        record, *_offsets(separation, angle), contrast=1e-10, time_d=0.0
    )
    column, row = expected(separation, angle, built)
    np.testing.assert_allclose(rates.star_leakage, column, rtol=rtol, atol=0)
    np.testing.assert_allclose(rates.local_zodi, row, rtol=rtol, atol=0)
    np.testing.assert_allclose(rates.exozodi, column, rtol=rtol, atol=0)
    # East is to the left and north is up at zero roll.
    assert built == 0.0 and column < centre < row
    rolled = cp.aperture_rates(
        record, *_offsets(separation, angle), contrast=1e-10, time_d=0.0, roll_deg=115.0
    )
    new_column, new_row = expected(separation, angle, 115.0)
    assert abs(new_column - column) > 1 and abs(new_row - row) > 1
    np.testing.assert_allclose(rolled.star_leakage, new_column, rtol=rtol, atol=0)
    np.testing.assert_allclose(rolled.local_zodi, new_row, rtol=rtol, atol=0)
    # The exozodi stays with the sky position it was rendered at.
    np.testing.assert_allclose(rolled.exozodi, column, rtol=rtol, atol=0)
    np.testing.assert_allclose(
        rolled.planet,
        rates.planet,
        rtol=roundoff(1, n_ops=4, dtype=rates.planet.dtype),
        atol=0,
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_local_zodi_follows_the_read_time(sloped):
    """Fails if a read keeps the build-time sightline or rescales another term.

    A target on the ecliptic is opposite the Sun at day 0, when the record is
    built, and 46 degrees from it at day 137, when the record is read and the
    Leinert radiance is several times higher. The independent reference is
    the image path: the aperture sum of the local-zodi images rendered at day
    137, which multiplies each pixel by the radiance before the sum where the
    read multiplies after it. A record rebuilt at day 137 and the radiance
    ratio of the two sightlines pin the contract: one record serves any time.
    Basis: floating-point. Against the image sum the difference is the
    rounding of two aperture reductions and 32 operations for weighting,
    the position read and radiance scaling. The radiance ratio and JIT
    comparisons each allow 32 operations for radiance interpolation,
    scaling and division.
    The rebuilt record holds the same uniform-sky table, so its read must be
    bit-identical in every term.
    """
    scene, observatory, _, _, response, _ = sloped
    scene = with_zodi(scene, LeinertZodi())
    scene = eqx.tree_at(lambda s: s.dynamics.frame.dec_deg, scene, jnp.asarray(23.0))
    built_at = [
        _build(scene, observatory, _BANDS, time_d=t, field_lod=0.0)
        for t in (0.0, 137.0)
    ]
    exposures = [exposure for exposure, _ in built_at]
    built, rebuilt = (
        cp.aperture_record(scene, observatory, exposure, response, sampling=sampling)
        for exposure, sampling in built_at
    )
    pitch = float(_sel(observatory).detector.pixel_scale_arcsec)
    x, y = 3.0 * pitch, -2.5 * pitch
    east, north = SkyOrientation(0.0).detector_to_sky(x, y)

    def read(record, time_d):
        return cp.aperture_rates(
            record,
            east,
            north,
            contrast=1e-10,
            time_d=time_d,
        )

    late, reference = read(built, 137.0), read(rebuilt, 137.0)
    photons = cp.photon_rates(scene, observatory, exposures[1], sampling=built_at[1][1])
    weights = _image_weights(observatory, x, y)
    rtol = roundoff(1, n_ops=32, dtype=late.local_zodi.dtype)
    sum_rtol = rtol + sum_roundoff(2.0, weights.size, dtype=late.local_zodi.dtype)
    for moment, expected, power in (
        (late, reference, 1),
        (late.variance, reference.variance, 2),
    ):
        assert np.all(moment.local_zodi > 0)
        np.testing.assert_allclose(
            moment.local_zodi,
            np.einsum("wqyx,yx->wq", np.asarray(photons.local_zodi), weights**power),
            rtol=sum_rtol,
            atol=0,
        )
        for name in ("planet", "star_leakage", "local_zodi", "exozodi", "n_pix"):
            np.testing.assert_array_equal(
                getattr(moment, name), getattr(expected, name)
            )
    # The time is seen, and by exactly the radiance ratio of the sightlines.
    nodes = built.quadrature.nodes_nm
    radiance = [leinert_radiance(scene, nodes, t) for t in (0.0, 137.0)]
    early = read(built, 0.0)
    assert np.all(abs(radiance[1] / radiance[0] - 1) > 0.5)
    np.testing.assert_allclose(
        late.local_zodi / early.local_zodi, radiance[1] / radiance[0], rtol=rtol, atol=0
    )
    # One record serves a batch of times under one compiled program.
    batched = jax.jit(jax.vmap(lambda t: read(built, t).local_zodi))(
        jnp.array([0.0, 137.0])
    )
    np.testing.assert_allclose(
        batched, jnp.stack([early.local_zodi, late.local_zodi]), rtol=rtol, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_read_under_jax_transformations(sloped, precision):
    """Fails if a batched, compiled or differentiated read departs from a plain one.

    Basis: vmap and JIT equal eager reads within 96 rounding steps for
    coordinate rotation, interpolation of both moments and S/N reduction.
    Central differences use float64, with the existing step and error
    bounds for separation and contrast.
    """
    record, detector = sloped[5], _sel(sloped[1]).detector

    def read(separation, angle, contrast):
        rates = cp.aperture_rates(
            record, *_offsets(separation, angle), contrast=contrast, time_d=0.0
        )
        return ox.snr(
            rates, detector, 3600.0, frame_time_s=100.0, post_processing=_NO_RESIDUAL
        )[0]

    separations = jnp.array([0.021, 0.04, 0.0633])
    angles = jnp.array([10.0, 200.0, 310.0])
    contrasts = jnp.array([1e-10, 3e-10, 2e-9])
    batched = jax.jit(jax.vmap(read))(separations, angles, contrasts)
    looped = jnp.stack(
        [read(*args) for args in zip(separations, angles, contrasts, strict=True)]
    )
    assert np.all(np.isfinite(looped))
    np.testing.assert_allclose(
        batched, looped, rtol=roundoff(1, n_ops=96, dtype=batched.dtype), atol=0
    )

    derivative = jax.grad(read, argnums=(0, 2))
    for gradient in (derivative, jax.jit(derivative)):
        slope, by_contrast = gradient(0.04, 200.0, 3e-10)
        assert np.isfinite(slope) and slope != 0
        assert np.isfinite(by_contrast) and by_contrast > 0
    if precision == np.float32:
        return

    step = 1e-7
    difference = (read(0.04 + step, 200.0, 3e-10) - read(0.04 - step, 200.0, 3e-10)) / (
        2 * step
    )
    np.testing.assert_allclose(slope, difference, rtol=1e-6)
    difference = (read(0.04, 200.0, 3.03e-10) - read(0.04, 200.0, 2.97e-10)) / 6e-12
    np.testing.assert_allclose(by_contrast, difference, rtol=1e-4)


@vv.case("aperture-photon-budget", "code-verification")
def test_reads_outside_the_tables_and_unsupported_builds(sloped):
    """Fails if a read outside the tables returns light or a bad build passes.

    Basis: exact. An unsupported read is zero in every field with ``supported``
    False, so ``snr`` is NaN and ``valid`` clears downstream, and a rule or
    quadrature the tables cannot represent raises at build time.
    """
    scene, observatory, _, sampling, response, record = sloped
    detector = _sel(observatory).detector
    last = float(response.separation_step_arcsec) * 8
    inside = cp.aperture_rates(
        record, *_offsets(0.5 * last, 0.0), contrast=1e-10, time_d=0.0
    )
    assert np.all(np.isfinite(inside.planet)) and np.all(np.isfinite(inside.n_pix))
    assert np.all(inside.supported) and np.all(inside.planet > 0)
    for separation in (1.01 * last, np.nan):
        outside = cp.aperture_rates(
            record, *_offsets(separation, 0.0), contrast=1e-10, time_d=0.0
        )
        assert not np.any(outside.supported)
        for field in (outside.planet, outside.star_leakage, outside.n_pix):
            assert np.all(np.asarray(field) == 0.0)
        assert np.isnan(
            ox.snr(
                outside,
                detector,
                3600.0,
                frame_time_s=100.0,
                post_processing=_NO_RESIDUAL,
            )
        ).all()
        solved = ox.exposure_time(
            outside, detector, 5.0, frame_time_s=100.0, post_processing=_NO_RESIDUAL
        )
        assert not solved.valid.any() and np.isnan(float(solved.time_s[0]))
    # The aperture leaves the detector before the pixel centre does.
    half = 24 * float(detector.pixel_scale_arcsec)
    edge = cp.aperture_rates(
        record, *_offsets(half * 0.99, 90.0), contrast=1e-10, time_d=0.0
    )
    assert not np.any(edge.supported)
    assert np.all(np.asarray(edge.star_leakage) == 0.0)
    assert np.all(np.asarray(edge.n_pix) == 0.0)

    for rule in (HalfMaximum(), Circular(0.7, centre="peak")):
        with pytest.raises(NotImplementedError, match="Circular"):
            cp.aperture_response(_sel(observatory), sampling.quadrature, rule=rule)
    other, other_sampling = _build(scene, observatory, (Band(510.0, 590.0),))
    with pytest.raises(StampMismatchError, match="layout"):
        cp.aperture_record(scene, observatory, other, response, sampling=other_sampling)
    unstamped = replace(response, built_at=())
    with pytest.raises(ValueError, match="other wavelength nodes"):
        cp.aperture_record(
            scene, observatory, other, unstamped, sampling=other_sampling
        )


def test_record_refuses_a_response_built_for_another_instrument(plateau):
    """Fails if a swapped coronagraph, pixel scale or band pairs with an old response.

    Basis: exact. Each swap keeps the wavelength nodes of the response and
    changes one thing: the coronagraph tables (a Gaussian for the flat PSF,
    seen only by the measured source), the detector pixel scale (seen in the
    aperture radius) and the band edges (seen in the bands). The observatory
    the response was built for passes.
    """
    scene, observatory, exposure, sampling, response, _ = plateau
    cp.aperture_record(scene, observatory, exposure, response, sampling=sampling)
    swapped = _instrument(GaussianCoronagraph(40, 0.25))
    with pytest.raises(ValueError, match="another coronagraph or detector"):
        cp.aperture_record(scene, swapped, exposure, response, sampling=sampling)
    detector = _sel(observatory).detector
    wider = copy.copy(detector)
    object.__setattr__(
        wider, "pixel_scale_arcsec", 1.1 * float(detector.pixel_scale_arcsec)
    )
    coarse = eqx.tree_at(lambda o: o.instruments[0].detector, observatory, wider)
    with pytest.raises(ValueError, match="pixel scale"):
        cp.aperture_record(scene, coarse, exposure, response, sampling=sampling)
    shifted = copy.copy(response.quadrature)
    object.__setattr__(shifted, "bands", (Band(500.0, 600.0001),))
    other = eqx.tree_at(lambda r: r.quadrature, response, shifted)
    with pytest.raises(ValueError, match="other bands"):
        cp.aperture_record(scene, observatory, exposure, other, sampling=sampling)


def test_flux_scale_scales_every_record_table_and_passes_the_guard(plateau):
    """Fails if the flux scale misses a table or invalidates the response.

    Basis: exact algebra, 1e-6 relative float32 rounding allowance. The scale
    acts once, in the effective area, so every rate table of a record built
    with scale 1.5 is 1.5 times the unit-scale table from the same response,
    and the pixel counts are unchanged.
    """
    scene, observatory, exposure, sampling, response, record = plateau
    scaled = eqx.tree_at(
        lambda o: o.instruments[0].flux_scale, observatory, jnp.asarray(1.5)
    )
    other = cp.aperture_record(scene, scaled, exposure, response, sampling=sampling)
    for name in ("planet_response", "star_leakage", "uniform_sky", "exozodi"):
        np.testing.assert_allclose(
            getattr(other, name), 1.5 * getattr(record, name), rtol=1e-6
        )
    np.testing.assert_array_equal(other.n_pix, record.n_pix)


@pytest.mark.parametrize(
    "leaf, planet_factor, leakage_factor",
    [("star", 1.0, 2.0), ("planet", 2.0, 1.0)],
)
def test_thermal_planet_scale_is_its_own_on_the_record_route(
    plateau, leaf, planet_factor, leakage_factor
):
    """Fails if a star's ``flux_scale`` reaches a thermal planet on the record route.

    Basis: a 300 K blackbody planet's flux does not depend on the host, so a
    star scale of two doubles only the leakage and a planet scale of two
    doubles only the planet, each by its factor within rtol=1e-5; the record
    is rebuilt for each scene.
    """
    _, observatory, _, _, response, _ = plateau
    planet = Planet(jnp.ones(1), BlackbodyPhysicalModel(teff_K=jnp.array([300.0])))
    scene = make_system(
        star=BlackbodyStar(5772.0, Rsun2m),
        separation_arcsec=0.03,
        planet_groups=(planet,),
    )
    where = {
        "star": lambda s: s.star.flux_scale,
        "planet": lambda s: s.planets[0].flux_scale,
    }[leaf]

    def read(scene):
        exposure, sampling = _build(scene, observatory, _BANDS, field_lod=0.0)
        record = cp.aperture_record(
            scene, observatory, exposure, response, sampling=sampling
        )
        return cp.scene_aperture_rates(record, scene, jnp.array([0.0]))

    base = read(scene)
    scaled = read(eqx.tree_at(where, scene, jnp.full_like(where(scene), 2.0)))
    assert float(jnp.max(base.planet)) > 0
    np.testing.assert_allclose(scaled.planet, planet_factor * base.planet, rtol=1e-5)
    np.testing.assert_allclose(
        scaled.star_leakage, leakage_factor * base.star_leakage, rtol=1e-5
    )


def test_scene_route_is_a_vmap_of_the_read_and_feeds_the_calculator(plateau):
    """Fails if the scene route differs from per-planet reads or mishandles axes.

    Basis: exact equality (0 ULP) with ``aperture_rates`` called planet by
    planet at positions and contrasts taken from the scene, an invalid planet
    reading zero in every field, and the calculator returning one value per
    band, planet and time from the returned ``Rates``.
    """
    _, observatory, exposure, sampling, response, _ = plateau
    scene = make_system(
        separation_arcsec=jnp.array([0.012, 0.03]),
        planet_groups=(
            Planet(
                jnp.ones(2),
                GridPhysicalModel(
                    jnp.array([400.0, 500.0, 600.0, 700.0]),
                    jnp.array([0.0, 90.0, 180.0]),
                    jnp.full((2, 4, 3), 1e-10),
                ),
            ),
        ),
    )
    scene = eqx.tree_at(lambda s: s.valid, scene, jnp.array([True, False]))
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    times = jnp.array([0.0, 30.0, 90.0])
    rates = cp.scene_aperture_rates(record, scene, times)
    nodes = record.quadrature.nodes_nm
    assert nodes.shape == (1, 4)
    assert rates.planet.shape == (1, 4, 2, 3) and rates.n_pix.shape == (1, 2, 3)
    assert rates.variance.planet.shape == (1, 4, 2, 3)
    assert rates.quadrature is record.quadrature and rates.carrier == "photon"
    east, north = scene.positions(times)
    for t, time in enumerate(times):
        contrast = jnp.array(
            [
                [float(scene.contrasts(w, jnp.atleast_1d(time))[0, 0]) for w in row]
                for row in np.asarray(nodes)
            ]
        )
        single = cp.aperture_rates(
            record, east[0, t], north[0, t], contrast=contrast, time_d=time
        )
        for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
            np.testing.assert_array_equal(
                getattr(rates, name)[..., 0, t], getattr(single, name)
            )
            np.testing.assert_array_equal(
                getattr(rates.variance, name)[..., 0, t],
                getattr(single.variance, name),
            )
        np.testing.assert_array_equal(rates.n_pix[:, 0, t], single.n_pix)
    for field in (rates.planet, rates.star_leakage, rates.n_pix, rates.exozodi):
        assert np.all(np.asarray(field[..., 1, :]) == 0.0)
    solved = ox.exposure_time(
        rates,
        _sel(observatory).detector,
        7.0,
        frame_time_s=100.0,
        post_processing=_NO_RESIDUAL,
    )
    assert solved.time_s.shape == solved.valid.shape == (1, 2, 3)
    assert np.all(np.asarray(solved.valid[:, 0])) and not solved.valid[:, 1].any()
    assert np.all(np.isnan(np.asarray(solved.time_s[:, 1])))
    with pytest.raises(ValueError, match="shape"):
        cp.scene_aperture_rates(record, scene, 0.0)


def _warm(observatory, temperature_K=290.0, emissivity=0.5):
    """The observatory with a lumped warm-telescope emission."""
    return eqx.tree_at(
        lambda p: p.instruments[0].warm_emission,
        observatory,
        ox.WarmEmission(temperature_K, emissivity),
        is_leaf=lambda x: x is None,
    )


def _pixel_thermal_rate(observatory, nodes_nm, temperature_K, emissivity):
    """Warm emission per detector pixel in photons/s/nm, from primitives.

    Planck photon radiance 2 c / lambda**4 / expm1(h c / (lambda k T)) per
    meter of wavelength, converted to per nanometer, times the emissivity, the
    geometric collecting area and the pixel solid angle.
    """
    wavelength_m = np.asarray(nodes_nm, dtype=np.float64) * nm2m
    x = h * c / (wavelength_m * k_B * temperature_K)
    radiance = 2 * c / wavelength_m**4 / np.expm1(x) * nm2m
    pixel_sr = (float(_sel(observatory).detector.pixel_scale_arcsec) * arcsec2rad) ** 2
    return emissivity * radiance * float(
        observatory.telescope.pupil.collecting_area_m2
    ) * pixel_sr, x


def test_thermal_record_and_scene_read(plateau):
    """Fails if the record drops, double counts or misshapes warm emission.

    Basis: the per-pixel rate from primitives times the aperture pixel count,
    for both aperture moments. The relative floor allows two rounding steps
    in the exponential's argument, amplified by its magnitude, and 24 for
    the prefactor, radiance and aperture multiplication. The scalar and scene
    reads agree exactly, an invalid planet reads zero; an observatory
    without warm emission reads exact zeros.
    """
    _, cold, exposure, sampling, response, _ = plateau
    observatory = _warm(cold)
    scene = make_system(
        separation_arcsec=jnp.array([0.012, 0.03]),
        planet_groups=(
            Planet(
                jnp.ones(2),
                GridPhysicalModel(
                    jnp.array([400.0, 500.0, 600.0, 700.0]),
                    jnp.array([0.0, 90.0, 180.0]),
                    jnp.full((2, 4, 3), 1e-10),
                ),
            ),
        ),
    )
    scene = eqx.tree_at(lambda s: s.valid, scene, jnp.array([True, False]))
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    pixel, exponent = _pixel_thermal_rate(
        observatory, record.quadrature.nodes_nm, 290.0, 0.5
    )
    rtol = roundoff(1, n_ops=2 * exponent.max() + 24, dtype=record.thermal.dtype)
    assert record.thermal.shape == (1, 4)
    np.testing.assert_allclose(record.thermal, pixel, rtol=rtol, atol=0)
    times = jnp.array([0.0, 30.0, 90.0])
    rates = cp.scene_aperture_rates(record, scene, times)
    assert rates.thermal.shape == (1, 4, 2, 3)
    assert rates.variance.thermal.shape == (1, 4, 2, 3)
    east, north = scene.positions(times)
    for t, time in enumerate(times):
        single = cp.aperture_rates(
            record, east[0, t], north[0, t], contrast=1e-10, time_d=time
        )
        np.testing.assert_allclose(
            single.thermal, pixel * np.asarray(single.n_pix[:, None]), rtol=rtol, atol=0
        )
        np.testing.assert_allclose(
            single.variance.thermal,
            pixel * np.asarray(single.variance.n_pix[:, None]),
            rtol=rtol,
            atol=0,
        )
        np.testing.assert_array_equal(rates.thermal[..., 0, t], single.thermal)
        np.testing.assert_array_equal(
            rates.variance.thermal[..., 0, t], single.variance.thermal
        )
    assert np.all(np.asarray(rates.thermal[..., 1, :]) == 0.0)
    assert np.all(np.asarray(rates.thermal[..., 0, :]) > 0.0)
    off = cp.aperture_rates(record, 1.0, 1.0, contrast=1e-10, time_d=0.0)
    assert not np.any(off.supported)
    np.testing.assert_array_equal(off.thermal, 0.0)
    cold_record = cp.aperture_record(scene, cold, exposure, response, sampling=sampling)
    np.testing.assert_array_equal(cold_record.thermal, 0.0)
    cold_rates = cp.scene_aperture_rates(cold_record, scene, times)
    np.testing.assert_array_equal(cold_rates.thermal, 0.0)


@pytest.mark.medium(reason="4096 detector frames of 48 by 48 pixels drawn on the host")
@vv.case("aperture-photon-budget", "code-verification")
def test_variance_matches_weighted_pixel_draws(sloped):
    """Fails if the w**2 moments or pixel counts misstate the weighted-sum variance.

    Pixels of the expected image are drawn as Poisson source, dark and
    clock-induced counts plus Gaussian read noise (seed 0), summed with the
    aperture weights, and the sample variance is compared with the record's.
    Basis: the draws confirm the weighted-sum formula within four standard
    errors of the sample variance, from the fourth Poisson cumulant and the
    variance of the unbiased estimator. The record then matches the formula
    within its interpolation between half-pixel positions and ring-mean
    planet rate: measured 3.5e-3 on the mean and 9.5e-3 on the variance,
    held to 1e-2 and 2e-2.
    """
    scene, observatory, exposure, sampling, response, _ = sloped
    base = _sel(observatory).detector
    detector = Detector(
        shape=base.shape,
        pixel_scale_arcsec=base.pixel_scale_arcsec,
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=2e-3,
        clock_induced_charge_e_per_pixel_frame=0.01,
        read_noise_e_rms=0.3,
    )
    observatory = eqx.tree_at(
        lambda p: p.instruments[0].detector, observatory, detector
    )
    bright = eqx.tree_at(lambda s: s.star.flux_phot_per_nm_m2, scene, jnp.asarray(1e9))
    # The response holds no detector noise, so the fixture's one serves.
    record = cp.aperture_record(
        bright, observatory, exposure, response, sampling=sampling
    )
    east, north = bright.positions(jnp.array([0.0]))[:2, 0, 0]
    x, y = SkyOrientation(0.0).sky_to_detector(east, north)
    weights = np.asarray(_image_weights(observatory, x, y))
    rates = cp.aperture_rates(
        record,
        east,
        north,
        contrast=1e-7,
        time_d=0.0,
    )
    plan = FramePlan(50.0, 4)
    mean, variance = detector.mean_and_variance(detector.electrons(rates), plan)

    images = cp.expected(bright, observatory, exposure, sampling=sampling)
    planet = np.asarray(images.planet[0]) * 1e-7 / 1e-10
    source = planet + np.asarray(images.star_leakage[0])
    poisson = (source + 2e-3) * 200.0 + 0.01 * 4
    draws = 4096
    rng = np.random.default_rng(0)
    counts = rng.poisson(poisson, (draws, *poisson.shape)) + rng.normal(
        0.0, 0.3 * 2.0, (draws, *poisson.shape)
    )
    sums = np.sum(counts * weights, axis=(1, 2))
    expected = np.sum(weights**2 * (poisson + 0.3**2 * 4))
    cumulant4 = np.sum(weights**4 * poisson)
    standard_error = np.sqrt(cumulant4 / draws + 2 * expected**2 / (draws - 1))
    assert abs(np.var(sums, ddof=1) - expected) < 4 * standard_error
    np.testing.assert_allclose(mean[0], np.sum(weights * poisson), rtol=1e-2)
    np.testing.assert_allclose(variance[0], expected, rtol=2e-2)
    # Counting whole pixels (weights w, not w**2) lands outside that band.
    assert np.sum(weights * (poisson + 0.3**2 * 4)) > 1.04 * expected


def test_aperture_read_needs_a_position_only_for_a_pointing_dependent_zodi(plateau):
    """An unplaced scene records and reads with AYOZodi; LeinertZodi refuses by name.

    The record holds the zodi model and the frame, and the model alone decides
    whether it needs a position, so the refusal comes from the read.
    """
    _, observatory, _, sampling, response, _ = plateau
    scene = make_system(separation_arcsec=0.03)
    scene = eqx.tree_at(
        lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
        scene,
        (None, None),
        is_leaf=lambda x: x is None,
    )
    exposure, sampling = _build(scene, observatory, _BANDS, field_lod=0.0)
    record = cp.aperture_record(
        with_zodi(scene, AYOZodi()), observatory, exposure, response, sampling=sampling
    )
    assert record.sky_frame.ra_deg is None
    rates = cp.aperture_rates(record, *_offsets(0.02, 0.0), contrast=1e-10, time_d=5.0)
    assert float(jnp.sum(rates.local_zodi)) > 0
    pointed = cp.aperture_record(
        with_zodi(scene, LeinertZodi()),
        observatory,
        exposure,
        response,
        sampling=sampling,
    )
    with pytest.raises(ValueError, match="no sky position"):
        cp.aperture_rates(pointed, *_offsets(0.02, 0.0), contrast=1e-10, time_d=5.0)


@vv.case("aperture-photon-budget", "code-verification")
def test_records_of_two_targets_stack_and_read_under_vmap(sloped):
    """Fails if a record carries static per-target data a stack cannot hold.

    Basis: two targets of one shape differ only in sky position, so their
    records stack leaf by leaf and one ``vmap`` over the stack equals the two
    separate reads within 64 rounding steps for the position, bilinear
    background reads and local-zodi coordinates and interpolation. The local zodi
    follows the pointing, so the two reads must also differ.
    """
    scene, observatory, exposure, sampling, response, _ = sloped
    scene = with_zodi(scene, LeinertZodi())
    other = eqx.tree_at(
        lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
        scene,
        (jnp.asarray(250.0), jnp.asarray(35.0)),
    )
    records = [
        cp.aperture_record(target, observatory, exposure, response, sampling=sampling)
        for target in (scene, other)
    ]
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *records)
    east, north = _offsets(0.04, 60.0)

    def read(record):
        return cp.aperture_rates(record, east, north, contrast=1e-10, time_d=60.0)

    together = eqx.filter_vmap(read)(stacked)
    for i, record in enumerate(records):
        single = read(record)
        for name in ("planet", "star_leakage", "local_zodi", "exozodi", "n_pix"):
            np.testing.assert_allclose(
                getattr(together, name)[i],
                getattr(single, name),
                rtol=roundoff(1, n_ops=64, dtype=getattr(single, name).dtype),
                atol=0,
            )
    zodi = np.asarray(together.local_zodi)
    assert np.all(abs(zodi[0] / zodi[1] - 1) > 0.01)


def test_record_builds_under_jit_and_jit_vmap_with_the_instrument_closed_over(sloped):
    """Fails if the response guard breaks a jitted record build.

    The guard measures concrete values at trace time even inside ``jax.jit``.
    Basis: floating-point, at each eager result's magnitude. Planet response
    allows four gain multiplications. Pixel counts sum the kernel weights,
    and leakage sums their products with the rendered pixels. Allow one
    rounding step per kernel tap for either convolution order, including
    separate products or fused multiply-adds. Leakage budgets 32 more steps
    for the wavelength and pixel coordinate conversions, the bilinear
    weights and up to nine native-pixel deposits per detector pixel, and
    scaling the sloped map to photon rates. With no disk, exozodi is exactly
    zero.
    """
    scene, observatory, exposure, sampling, response, _ = sloped
    scene = with_zodi(scene, LeinertZodi())

    def build(ra_deg, dec_deg):
        pointed = eqx.tree_at(
            lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
            scene,
            (ra_deg, dec_deg),
        )
        return cp.aperture_record(
            pointed, observatory, exposure, response, sampling=sampling
        )

    ra, dec = jnp.asarray([250.0, 260.0]), jnp.asarray([35.0, 40.0])
    eager = [build(ra[i], dec[i]) for i in range(2)]
    jitted = jax.jit(build)(ra[0], dec[0])
    batched = jax.jit(jax.vmap(build))(ra, dec)
    kernel_taps = response.kernels.shape[-2] * response.kernels.shape[-1]
    for name, n_ops in (
        ("planet_response", 4),
        ("star_leakage", kernel_taps + 32),
        ("exozodi", 0),
        ("n_pix", kernel_taps),
    ):
        dtype = getattr(eager[0], name).dtype
        rtol = float(roundoff(1, n_ops=n_ops, dtype=dtype))  # tol: algorithm, roundoff
        np.testing.assert_allclose(
            getattr(jitted, name), getattr(eager[0], name), rtol=rtol, atol=0
        )
        for i in range(2):
            np.testing.assert_allclose(
                getattr(batched, name)[i], getattr(eager[i], name), rtol=rtol, atol=0
            )
    # Static facts equal the eager ones, no tracer reached the record, and the view is
    # not a field.
    for record in (jitted, batched):
        assert record.built_at == eager[0].built_at
        assert record.reads == eager[0].reads
        assert record.epoch == eager[0].epoch
        assert not any(
            isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(record)
        )
    assert not any(
        isinstance(getattr(eager[0], f.name), ox.SelectedInstrument)
        for f in dataclasses.fields(eager[0])
    )

    # One compilation serves two commanded times and two rolls, because the stamps
    # are facts of the hardware and the layout and not of any value.
    traces = []

    def build_at(time_d, roll_deg):
        traces.append(1)
        moved = replace(exposure, time_d=time_d, pointing=ox.Pointing(roll_deg))
        return cp.aperture_record(
            scene, observatory, moved, response, sampling=sampling
        )

    compiled = jax.jit(build_at)
    first = compiled(jnp.asarray(0.0), jnp.asarray(0.0))
    second = compiled(jnp.asarray(3.0), jnp.asarray(25.0))
    assert len(traces) == 1
    assert first.build_time_d == 0.0 and second.build_time_d == 3.0
    assert second.pointing.roll_deg == 25.0
    assert second.built_at == first.built_at


def test_response_throughput_matches_stored_curve(tmp_path):
    """The aperture response and the YIP throughput curve describe one coronagraph.

    The response integrates the off-axis PSF inside the rule's aperture on the
    detector; the stored curve is the throughput yippy computes from the same
    PSFs for the same rule. Averaged over position angle they agree to the
    response's sampling error, set by the 6 rings of 8 angles and the detector
    pixel grid. Measured here: 2.4e-3 relative and 2.8e-4 absolute at most; the
    bound below leaves a factor of about eight of margin.
    """
    rule = Circular(0.7)
    coro = open_yip(write_yip(tmp_path / "yip")).prepare(rules=(rule,))
    scale = float(coro.frame.lod_to_arcsec(550.0))
    observatory = observatory_for(
        SimplePupil(
            circumscribed_diameter_m=coro.frame.diameter_m, collecting_area_m2=10.0
        ),
        (ConstantThroughput(throughput=1.0),),
        coro,
        Detector(pixel_scale_arcsec=0.1 * scale, shape=(96, 96), qe=1.0),
    )
    _, sampling = _build(make_system(), observatory, (Band(549.0, 551.0),))
    response = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        rule=rule,
        n_separations=6,
        n_angles=8,
        max_separation_arcsec=3.0 * scale,
    )
    n_sep = response.throughput.shape[-1]
    sep_lod = np.arange(n_sep) * response.separation_step_arcsec / scale
    expected = np.asarray(coro.curves().evaluate("throughput", sep_lod))
    got = np.asarray(response.throughput)[0, 0].mean(axis=0)
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=7e-3)


@vv.case("aperture-photon-budget", "code-verification")
def test_reads_beyond_the_stamp_are_zero_and_unsupported():
    """Fails if an aperture on unmodeled pixels reads light or is not flagged.

    Detector twice the table (48 pixels of 0.25 lambda/D, table half-width
    3 lambda/D). An aperture of radius 0.7 lambda/D centred inside the stamp
    reads positive star leakage with ``supported`` True; centred at 4.5
    lambda/D, wholly beyond the stamp but on the detector, every field is
    exactly zero with ``supported`` False. A grid of centres across the stamp
    edge is supported exactly when its aperture lies within the 2.875
    lambda/D the linear products cover. Basis: exact.
    """
    from conftest import GaussianCoronagraph

    observatory = _instrument(GaussianCoronagraph(24, 0.25))
    frame = _sel(observatory).coronagraph.grid.frame
    observatory = eqx.tree_at(
        lambda p: p.instruments[0].detector,
        observatory,
        Detector(
            pixel_scale_arcsec=0.25 * float(frame.lod_to_arcsec(550.0)),
            shape=(48, 48),
            qe=1.0,
        ),
    )
    scale = float(frame.lod_to_arcsec(550.0))
    scene = make_system(separation_arcsec=1.0 * scale)
    exposure, sampling = _build(scene, observatory, _BANDS)
    response = cp.aperture_response(
        _sel(observatory), sampling.quadrature, n_separations=21, n_angles=8
    )
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )

    def read(separation_lod):
        return cp.aperture_rates(
            record, separation_lod * scale, 0.0, contrast=1e-10, time_d=0.0
        )

    near = read(1.0)
    assert bool(near.supported.all()) and np.all(np.asarray(near.star_leakage) > 0)
    far = read(4.5)
    assert not np.any(far.supported)
    for name in ("planet", "star_leakage", "local_zodi", "exozodi", "thermal", "n_pix"):
        np.testing.assert_array_equal(getattr(far, name), 0.0)
        np.testing.assert_array_equal(getattr(far.variance, name), 0.0)
    # The aperture is inside the support while centre + 0.7 <= 2.875, at every
    # node (the footprint at the blue node of the band is the binding one).
    blue = float(frame.lod_to_arcsec(sampling.quadrature.nodes_nm.min()))
    limit = (2.875 * blue / scale) - 0.7
    centres = np.linspace(0.5, 3.8, 12)
    flags = np.array([bool(read(c).supported.all()) for c in centres])
    np.testing.assert_array_equal(flags, centres <= limit + 1e-9)
    assert flags.any() and not flags.all()


@pytest.mark.medium(reason="YIP load and a one-node, 12-ring response")
@vv.case("aperture-photon-budget", "code-verification")
def test_planet_beyond_the_hull_reads_zero_and_unsupported(tmp_path):
    """Fails if a planet beyond the coronagraph's source nodes reads light.

    A synthetic YIP whose source nodes fill a square of half-width 3 lambda/D
    (its corner is 4.2) on a detector that holds an aperture at 5 lambda/D,
    still inside the 6 lambda/D stamp the pixels cover. Before, the record
    route read the planet's tabulated ring as zero without a flag; the planet
    is now unsupported there alone: its ring is flagged, every field is zero,
    and a planet at 2.6 lambda/D on the same record reads positive and
    supported. A ring is supported when every azimuth of it is inside the
    nodes, so the table flags rings by the hull's inscribed radius. Basis:
    exact.
    """
    coro = open_yip(write_yip(tmp_path / "yip")).prepare(
        rules=(_RULE,), leakage_rule="aperture_mean"
    )
    scale = float(coro.frame.lod_to_arcsec(550.0))
    observatory = observatory_for(
        SimplePupil(
            circumscribed_diameter_m=coro.frame.diameter_m, collecting_area_m2=10.0
        ),
        (ConstantThroughput(throughput=1.0),),
        coro,
        Detector(pixel_scale_arcsec=0.2 * scale, shape=(64, 64), qe=1.0),
    )
    scene = make_system(separation_arcsec=2.6 * scale)
    exposure, sampling = _build(
        scene, observatory, (Band(549.0, 551.0),), field_lod=0.0, order=1
    )
    response = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        n_separations=12,
        n_angles=8,
        max_separation_arcsec=5.5 * scale,
    )
    rings = np.asarray(response.throughput_supported)[0, 0]
    step_lod = float(response.separation_step_arcsec) / scale
    np.testing.assert_array_equal(rings, np.arange(12) * step_lod <= 3.0)
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    inner = cp.aperture_rates(record, 2.6 * scale, 0.0, contrast=1e-10, time_d=0.0)
    outer = cp.aperture_rates(record, 5.0 * scale, 0.0, contrast=1e-10, time_d=0.0)
    assert bool(inner.supported.all()) and np.all(np.asarray(inner.planet) > 0)
    assert not np.any(outer.supported)
    for name in ("planet", "star_leakage", "local_zodi", "n_pix"):
        np.testing.assert_array_equal(getattr(outer, name), 0.0)


@pytest.mark.medium(reason="one 21-ring response on a 48 pixel detector, 6 s cold")
@vv.case("aperture-photon-budget", "code-verification")
def test_curves_and_record_routes_agree_on_the_flag_across_the_stamp_edge():
    """Fails if the curves route and the record route disagree on what is supported.

    Seam anchor, ``jaxedith.rates`` against ``cp.aperture_rates`` on the same
    observatory: a detector twice the table (table half-width 3 lambda/D) and a
    2 nm band, so the band-centre read of the curves and the blue node of the
    record share one footprint. On the +x axis the five separations 0.5, 1.5,
    2.0, 2.6 and 3.5 lambda/D give the same ``supported`` on both routes, and
    both zero every field where it is False. The routes differ by design in
    two places, asserted as such. The curves are supported while separation
    plus the rule's radius is within the table's half-width, a disk; the
    record's footprint is the square stamp shrunk by the half source pixel the
    linear products need, so at 2.25 lambda/D on the axis the
    curves are True and the record is False. On the diagonal the square reaches
    farther than the disk: at (2.0, 2.0) lambda/D (separation 2.83) the record
    is True and the curves are False. The values are not compared (the curves
    read an azimuthal leakage mean, the record the value at the planet).
    Basis: exact.
    """
    from jaxedith import rates as curve_rates

    observatory = _instrument(GaussianCoronagraph(24, 0.25))
    scale = float(_sel(observatory).coronagraph.grid.frame.lod_to_arcsec(550.0))
    observatory = eqx.tree_at(
        lambda p: p.instruments[0].detector,
        observatory,
        Detector(pixel_scale_arcsec=0.25 * scale, shape=(48, 48), qe=1.0),
    )
    scene = make_system(separation_arcsec=1.0 * scale)
    exposure, sampling = _build(scene, observatory, (Band(549.0, 551.0),))
    response = cp.aperture_response(
        _sel(observatory), sampling.quadrature, n_separations=21, n_angles=8
    )
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    selected = _sel(observatory)
    curves = selected.coronagraph.curves()
    sources = scene.sources(sampling.quadrature, sampling.resolution, jnp.array([0.0]))

    def flags(east_lod, north_lod):
        east, north = east_lod * scale, north_lod * scale
        image = cp.aperture_rates(record, east, north, contrast=1e-10, time_d=0.0)
        analytic = curve_rates(
            curves,
            scene,
            selected,
            sampling.quadrature,
            east,
            north,
            contrast=1e-10,
            time_d=0.0,
            sources=sources,
        )
        for rates in (image, analytic):
            if not np.any(rates.supported):
                for name in ("planet", "star_leakage", "n_pix"):
                    np.testing.assert_array_equal(getattr(rates, name), 0.0)
        return bool(image.supported.all()), bool(analytic.supported.all())

    for separation, expected in zip(
        (0.5, 1.5, 2.0, 2.6, 3.5), (True, True, True, False, False), strict=True
    ):
        assert flags(separation, 0.0) == (expected, expected), separation
    assert flags(2.25, 0.0) == (False, True)
    assert flags(2.0, 2.0) == (True, False)
    assert flags(3.0, 3.0) == (False, False)


class _CentralField(ox.AbstractSpeckleField):
    """A uniform excess on 16 cells of 0.25 lambda/D, narrower than the table."""

    grid: PixelGrid = eqx.field(
        static=True,
        default=PixelGrid.focal(16, 0.25, Frame(550.0, 6.0, "circumscribed")),
    )
    epoch: Epoch = eqx.field(static=True, default=Epoch(2451545.0))
    epoch_d: float = eqx.field(static=True, converter=float, default=0.0)

    def spectral_breaks_nm(self):
        """The excess is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        """Ten times the static floor in every cell."""
        return Image(
            jnp.full(self.grid.shape, 1e-8 * self.grid.pixel_area),
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )


def test_aperture_beyond_the_field_reads_zero_inside_the_static_table():
    """Guard: fails if static light beyond a speckle field leaks into an aperture.

    A field reaching 2 lambda/D inside a table reaching 5 lambda/D. An aperture
    of radius 0.7 lambda/D centred at about 3.2 lambda/D lies inside the table
    and wholly beyond the field: without the field it is supported and reads
    starlight; with it, it reads zero in every field with ``supported`` False.
    Truth: the strict support rule; exact.
    """
    scene = make_system(separation_arcsec=0.03)
    base = _instrument(GaussianCoronagraph(40, 0.25))
    exposure, sampling = _build(scene, base, _BANDS, field_lod=0.0)
    response = cp.aperture_response(
        _sel(base),
        sampling.quadrature,
        n_separations=5,
        n_angles=4,
        max_separation_arcsec=0.08,
    )
    with_field = eqx.tree_at(
        lambda p: p.instruments[0].speckle,
        base,
        _CentralField(),
        is_leaf=lambda v: v is None,
    )
    offsets = _offsets(0.06, 90.0)
    static = cp.aperture_rates(
        cp.aperture_record(scene, base, exposure, response, sampling=sampling),
        *offsets,
        contrast=1e-10,
        time_d=0.0,
    )
    assert np.all(static.supported) and np.all(np.asarray(static.star_leakage) > 0)
    rates = cp.aperture_rates(
        cp.aperture_record(scene, with_field, exposure, response, sampling=sampling),
        *offsets,
        contrast=1e-10,
        time_d=0.0,
    )
    assert not np.any(rates.supported)
    for name in (*rates.component_names(), "n_pix"):
        assert np.all(np.asarray(getattr(rates, name)) == 0.0), name


@pytest.mark.slow(
    reason="three-band and single-band aperture records, ~20 s projected in CI"
)
def test_aperture_tables_image_the_real_nodes_only(monkeypatch):
    """Fails if the response or record images padded slots, or the padding moves them.

    Three bands split at the local zodi's knots hold different numbers of real
    nodes. Truth: the response images ``n_separations`` rings at each real
    node plus one probe per band, at the band's middle real node, and the
    record sums three maps (star, uniform sky, disk) at each real node (counts
    and probe wavelengths exact, from the static layout). A band built alone
    has no padding, so its tables and probe must equal the joint ones of that
    band (tolerance 1e-12 of each table's peak, basis floating point on the
    same nodes in a different batch layout), and each padded slot must repeat
    the band's first real node bit for bit.
    """
    from coronagraphoto import aperture

    observatory = _instrument(PlateauCoronagraph(40, 0.25))
    scene = with_zodi(make_system(separation_arcsec=0.03), AYOZodi())
    bands = (Band(480.0, 520.0), Band(560.0, 600.0), Band(420.0, 498.0))
    options = dict(n_separations=3, n_angles=2, max_separation_arcsec=0.04)
    exposure, sampling = _build(scene, observatory, bands, field_lod=0.0)
    counts = [sum(n for *_, n in band) for band in sampling.quadrature.pieces]
    width = sampling.quadrature.nodes_nm.shape[1]
    assert len(set(counts)) > 1 and max(counts) == width

    imaged, summed = [], []
    ring_moments, lattice = aperture._ring_moments, aperture._lattice

    def counted_moments(model, detector, wavelengths_nm, *args):
        imaged.append(np.asarray(wavelengths_nm))
        return ring_moments(model, detector, wavelengths_nm, *args)

    def counted_lattice(maps, *args):
        summed.append(maps.shape[1])
        return lattice(maps, *args)

    monkeypatch.setattr(aperture, "_ring_moments", counted_moments)
    monkeypatch.setattr(aperture, "_lattice", counted_lattice)
    response = cp.aperture_response(_sel(observatory), sampling.quadrature, **options)
    assert sum(map(len, imaged)) == options["n_separations"] * sum(counts) + len(bands)
    nodes = np.asarray(sampling.quadrature.nodes_nm)
    probes = imaged[-len(bands) :]
    for i, n in enumerate(counts):
        np.testing.assert_array_equal(probes[i], nodes[i, n // 2 : n // 2 + 1])
    summed.clear()
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    # One sum per band of the three maps, then the pixel count and support.
    assert summed[: len(bands)] == [3 * n for n in counts]
    monkeypatch.undo()

    for i, band in enumerate(bands):
        alone, alone_sampling = _build(scene, observatory, (band,), field_lod=0.0)
        assert alone_sampling.quadrature.nodes_nm.shape[1] == counts[i]
        single = cp.aperture_response(
            _sel(observatory), alone_sampling.quadrature, **options
        )
        single_record = cp.aperture_record(
            scene, observatory, alone, single, sampling=alone_sampling
        )
        n = counts[i]
        pairs = (
            (response.throughput[:, i, :n], single.throughput[:, 0]),
            (record.star_leakage[:, i, :n], single_record.star_leakage[:, 0]),
            (record.uniform_sky[:, i, :n], single_record.uniform_sky[:, 0]),
            (response.probe[i], single.probe[0]),
        )
        for joint, truth in pairs:
            truth = np.asarray(truth)
            np.testing.assert_allclose(
                joint, truth, rtol=0, atol=1e-12 * np.max(np.abs(truth))
            )
        for table in (
            response.throughput[:, i],
            record.star_leakage[:, i],
            record.uniform_sky[:, i],
        ):
            for q in range(n, width):
                np.testing.assert_array_equal(table[:, q], table[:, 0])


class ConstantFluxModel(BlackbodyPhysicalModel):
    """Manufactured steady emitter, independent of host flux and wavelength."""

    def _photon_flux_density(self, wavelength_nm, *, phase_angle_rad, **kwargs):
        """Return three photons/s/m^2/nm for every planet and time."""
        return jnp.full_like(phase_angle_rad, 3.0)


@pytest.fixture(scope="module")
def unit_response(plateau):
    """Reuse the synthetic observatory with unit gain and table throughput."""
    _, original, _, _, response, _ = plateau
    observatory = observatory_for(
        SimplePupil(circumscribed_diameter_m=6.0, collecting_area_m2=1.0),
        (ConstantThroughput(1.0),),
        _sel(original).coronagraph,
        _sel(original).detector,
    )
    response = eqx.tree_at(
        lambda r: r.throughput, response, jnp.ones_like(response.throughput)
    )
    return observatory, response


@vv.case("aperture-photon-budget", "code-verification")
def test_zero_star_thermal_planet_flux(unit_response):
    """Truth: source photon density times unit gain and table throughput.

    Floating-point basis: rtol=1e-12, atol=0 for the x64 hand product.
    Dividing by the exactly zero star before multiplying must fail this test.
    """
    observatory, response = unit_response
    scene = make_system(
        star=FlatStar(flux_phot_per_nm_m2=1.0, flux_scale=0.0),
        separation_arcsec=0.03,
        planet_groups=(
            Planet(jnp.ones(1), BlackbodyPhysicalModel(jnp.array([300.0]))),
        ),
    )
    exposure, sampling = _build(scene, observatory, _BANDS, field_lod=0.0)
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    times = jnp.array([0.0])
    expected = scene.sources(sampling.quadrature, sampling.resolution, times).planets
    rates = cp.scene_aperture_rates(record, scene, times)
    assert np.isfinite(rates.planet).all()
    assert rates.supported.all()
    np.testing.assert_allclose(rates.planet, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rates.variance.planet, expected, rtol=1e-12, atol=0)


@vv.case("aperture-photon-budget", "code-verification")
@pytest.mark.medium(
    reason="Time-dependent spectral reduction and record read take over 5 s"
)
def test_variable_star_constant_planet_flux(unit_response):
    """Truth: a steady three-photon emitter on a star of flux one then two.

    Unit gain and throughput give [3, 3]. Allow eight work-dtype roundings
    at unit relative scale for unit conversion and spectral reduction, with
    the 1e-12 relative floor for x64. A build-time stellar factor gives [3, 1.5].
    """
    observatory, response = unit_response
    wave = jnp.array([400.0, 700.0])
    flux = jnp.array([[1.0, 2.0], [1.0, 2.0]])
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0, 1.0]),
        flux_density_jy=photons_per_nm_per_m2_to_jy(flux, wave[:, None]),
        interpolation_method="linear",
    )
    scene = make_system(
        star=star,
        separation_arcsec=0.03,
        planet_groups=(Planet(jnp.ones(1), ConstantFluxModel(jnp.array([300.0]))),),
    )
    exposure, sampling = _build(scene, observatory, _BANDS, field_lod=0.0)
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    times = jnp.array([0.0, 1.0])
    sources = scene.sources(sampling.quadrature, sampling.resolution, times)
    rtol = max(1e-12, float(ulps(8, jnp.ones((), dtype=sources.star.dtype))))
    np.testing.assert_allclose(
        sources.star,
        jnp.broadcast_to(jnp.array([1.0, 2.0]), sources.star.shape),
        rtol=rtol,
        atol=0,
    )
    np.testing.assert_allclose(sources.planets, 3.0, rtol=rtol, atol=0)
    rates = cp.scene_aperture_rates(record, scene, times)
    assert rates.supported.all()
    np.testing.assert_allclose(rates.planet, 3.0, rtol=rtol, atol=0)


@vv.case("aperture-photon-budget", "code-verification")
@pytest.mark.medium(reason="Two-band response build and x64 compilation take over 5 s")
def test_planet_flux_and_contrast_node_axes(unit_response):
    """Truth: reflected flux equals star times contrast at distinct (W=2,Q=4) nodes.

    Floating-point basis: rtol=1e-12, atol=0 for the two physical inputs.
    Exact-algebra basis for bitwise legacy arithmetic: unit gain/throughput
    make interpolation constant, so reassociation introduces no rounding.
    Flipping the stellar node axis changes the nonconstant spectral values.
    """
    observatory, _ = unit_response
    wave = jnp.array([400.0, 500.0, 600.0, 700.0])
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0]),
        flux_density_jy=photons_per_nm_per_m2_to_jy(
            jnp.array([1.0, 2.0, 4.0, 8.0]), wave
        )[:, None],
        interpolation_method="linear",
    )
    scene = make_system(star=star, separation_arcsec=0.03)
    exposure, sampling = _build(
        scene,
        observatory,
        (Band(500.0, 550.0), Band(550.0, 600.0)),
        field_lod=0.0,
    )
    response = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        n_separations=2,
        n_angles=1,
        max_separation_arcsec=0.04,
    )
    response = eqx.tree_at(
        lambda r: r.throughput, response, jnp.ones_like(response.throughput)
    )
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    sources = scene.sources(sampling.quadrature, sampling.resolution, jnp.array([0.0]))
    assert sources.star.shape[:2] == (2, 4)
    east, north = scene.positions(jnp.array([0.0]))
    kwargs = dict(time_d=0.0)
    flux = cp.aperture_rates(
        record, east[0, 0], north[0, 0], planet=sources.planets[..., 0, 0], **kwargs
    )
    ratio = cp.aperture_rates(
        record, east[0, 0], north[0, 0], contrast=sources.contrast[..., 0, 0], **kwargs
    )
    assert ratio.supported.all()
    np.testing.assert_allclose(flux.planet, ratio.planet, rtol=1e-12, atol=0)
    # Legacy table is gain * star * throughput, constant on separation rings.
    legacy_table = (jnp.ones_like(sources.star[..., 0]) * sources.star[..., 0])[
        None, :, :, None
    ] * response.throughput
    position = jnp.hypot(east[0, 0], north[0, 0]) / response.separation_step_arcsec
    legacy = legacy_table[..., 0] + position * (
        legacy_table[..., 1] - legacy_table[..., 0]
    )
    np.testing.assert_array_equal(
        ratio.planet, (sources.contrast[..., 0, 0] * legacy)[0]
    )


@vv.case("aperture-photon-budget", "code-verification")
@pytest.mark.parametrize("kwargs", [{}, {"contrast": 1e-10, "planet": 3.0}])
def test_aperture_requires_one_planet_input(plateau, kwargs):
    """Truth: one radiometric input; exact-algebra host refusal, no tolerance."""
    record = plateau[-1]
    for read in (cp.aperture_rates, jax.jit(cp.aperture_rates)):
        with pytest.raises(ValueError, match=r"contrast.*planet|planet.*contrast"):
            read(record, 0.03, 0.0, time_d=0.0, **kwargs)


@pytest.fixture(scope="module")
def two_band_record(unit_response):
    """Return a (W=2, Q=4) record and the read position, at unit throughput."""
    observatory, _ = unit_response
    wave = jnp.array([400.0, 500.0, 600.0, 700.0])
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0]),
        flux_density_jy=photons_per_nm_per_m2_to_jy(
            jnp.array([1.0, 2.0, 4.0, 8.0]), wave
        )[:, None],
        interpolation_method="linear",
    )
    scene = make_system(star=star, separation_arcsec=0.03)
    exposure, sampling = _build(
        scene,
        observatory,
        (Band(500.0, 550.0), Band(550.0, 600.0)),
        field_lod=0.0,
    )
    response = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        n_separations=2,
        n_angles=1,
        max_separation_arcsec=0.04,
    )
    response = eqx.tree_at(
        lambda r: r.throughput, response, jnp.ones_like(response.throughput)
    )
    record = cp.aperture_record(
        scene, observatory, exposure, response, sampling=sampling
    )
    east, north = scene.positions(jnp.array([0.0]))
    return record, east[0, 0], north[0, 0]


@vv.case("aperture-photon-budget", "code-verification")
@pytest.mark.medium(reason="Two-band response build and x64 compilation take over 5 s")
@pytest.mark.parametrize("keyword", ["planet", "contrast"])
def test_planet_input_with_an_extra_axis_is_refused(two_band_record, keyword):
    """Truth: an input is (W, Q) or broadcasts to it; a batch is not silently read.

    Exact-algebra host refusal, no tolerance. Without the shape check before
    the response product, a stacked pair becomes the mean and variance inputs.
    """
    record, east, north = two_band_record
    one = jnp.full(record.star.shape, 1e-10)
    batch = jnp.stack([one, 17 * one])
    for read in (cp.aperture_rates, jax.jit(cp.aperture_rates)):
        with pytest.raises(ValueError, match=keyword):
            read(record, east, north, time_d=0.0, **{keyword: batch})
    ok = cp.aperture_rates(record, east, north, time_d=0.0, **{keyword: one})
    assert ok.supported.all()


@vv.case("aperture-photon-budget", "code-verification")
@pytest.mark.medium(reason="Two-band response build and x64 compilation take over 5 s")
def test_contrast_read_promotes_to_the_response_dtype(two_band_record, precision):
    """Truth: float32 star and contrast use the response's working precision.

    Floating-point basis: the product of the float32 values taken exactly in
    float64 with four rounding steps for the products. The float64 lane
    also compares the narrowed response and coordinates with the wide
    read, allowing 24 steps for coordinate and ring interpolation.
    """
    record, east, north = two_band_record
    star = (record.star * (1 + 1 / 3)).astype(jnp.float32)
    contrast = jnp.full(record.star.shape, 1.1e-10 / 3, dtype=jnp.float32)
    record = eqx.tree_at(lambda r: r.star, record, star)
    assert record.planet_response.dtype == precision
    expected = np.asarray(star, np.float64) * np.asarray(contrast, np.float64)
    for read in (cp.aperture_rates, jax.jit(cp.aperture_rates)):
        rates = read(record, east, north, contrast=contrast, time_d=0.0)
        unit = cp.aperture_rates(record, east, north, planet=1.0, time_d=0.0)
        assert rates.planet.dtype == precision
        np.testing.assert_allclose(
            rates.planet,
            expected * np.asarray(unit.planet),
            rtol=roundoff(1, n_ops=4, dtype=precision),
            atol=0,
        )
    if precision == np.float64:
        precision_pair(
            lambda rec, e, n, c: (
                cp.aperture_rates(rec, e, n, contrast=c, time_d=0.0).planet
            ),
            (record, east, north, contrast),
            bound=24 * eps(np.float32),
            scale=0,
        )


@vv.case("aperture-photon-budget", "code-verification")
@pytest.mark.medium(reason="Two-band response build and x64 compilation take over 5 s")
def test_planet_response_keeps_precision_near_a_throughput_drop(two_band_record):
    """Truth: the weighted sum of the two adjacent ring rates, exactly rounded.

    Floating-point basis: the throughput falls from 0.41 to 8.7e-10 between
    the rings; the read just inside the outer ring is compared with the exact
    rational interpolation of the binary64 inputs. Six rounding steps
    allow narrowing the endpoint values and offset, a difference, multiply
    and addition. Admission at the exact host endpoint and just beyond it is
    Boolean and both lanes must retain the small inside offset.
    """
    record, _, _ = two_band_record
    high, low = 0.41, 8.7e-10
    table = jnp.stack(
        [jnp.full(record.star.shape, high), jnp.full(record.star.shape, low)], -1
    )
    record = eqx.tree_at(
        lambda r: r.planet_response,
        record,
        jnp.broadcast_to(table, record.planet_response.shape),
    )
    step = 0.04  # The response fixture's host endpoint, rounded down in float32.
    inside = step * (1 - 1e-9)
    outside = step * (1 + 1e-9)
    east = np.array([inside, step, outside])
    f = Fraction(inside) / Fraction(step)
    exact = float((1 - f) * Fraction(high) + f * Fraction(low))

    def check(flux, supported):
        np.testing.assert_array_equal(supported[:2], True)
        np.testing.assert_array_equal(supported[2], False)
        np.testing.assert_allclose(
            flux[0], exact, rtol=0, atol=roundoff(exact, n_ops=6, dtype=flux.dtype)
        )
        np.testing.assert_array_equal(flux[1], np.asarray(low, dtype=flux.dtype))
        np.testing.assert_array_equal(flux[2], 0)

    def read(record, *coordinates):
        rates = cp.aperture_rates(record, *coordinates, planet=1.0, time_d=0.0)
        return rates.planet, rates.supported

    host_reads = [read(record, x, 0.0) for x in east]
    flux, supported = zip(*host_reads, strict=True)
    check(np.stack(flux), np.stack(supported))
    positions = record.prepare_position(east, 0.0)
    missing_inner = eqx.tree_at(
        lambda r: r.planet_unsupported,
        record,
        record.planet_unsupported.at[..., 0].set(1),
    )
    batch = jax.vmap(read, in_axes=(None, 0))
    for evaluate in (batch, jax.jit(batch)):
        check(*evaluate(record, positions))
        flux, supported = evaluate(missing_inner, positions)
        np.testing.assert_array_equal(supported[0], False)
        np.testing.assert_array_equal(supported[1], True)
        np.testing.assert_array_equal(flux[0], 0)

    # Raw JAX arguments on either side of the edge round to the same
    # decision coordinate. The next float32 value outside must be rejected.
    adjacent = np.nextafter(np.float32(step), np.float32(np.inf))
    raw = jnp.asarray([*east, adjacent])
    raw_fraction = Fraction(float(raw[0])) / Fraction(
        float(record.separation_step_arcsec)
    )
    raw_truth = float(
        (1 - raw_fraction) * Fraction(high) + raw_fraction * Fraction(low)
    )
    batch = jax.vmap(read, in_axes=(None, 0, None))
    for evaluate in (batch, jax.jit(batch)):
        flux, supported = evaluate(record, raw, 0.0)
        np.testing.assert_allclose(
            flux[0],
            raw_truth,
            rtol=0,
            atol=roundoff(raw_truth, n_ops=6, dtype=flux.dtype),
        )
        np.testing.assert_array_equal(flux[1:3], np.asarray(low, dtype=flux.dtype))
        np.testing.assert_array_equal(flux[3], 0)
        np.testing.assert_array_equal(supported[:3], True)
        np.testing.assert_array_equal(supported[3], False)
        flux, supported = evaluate(missing_inner, raw, 0.0)
        np.testing.assert_array_equal(supported[:3], True)
        np.testing.assert_array_equal(flux[:3], np.asarray(low, dtype=flux.dtype))


def test_aperture_geometry_uses_the_child_at_the_band_center():
    """The 545 nm aperture uses the first child irrespective of other bands."""
    blue = Frame(550.0, 6.0, "circumscribed")
    red = Frame(550.0, 4.0, "inscribed")
    children = tuple(
        GaussianCoronagraph(n=24, bandpass=Band(lo, hi), frame=frame)
        for lo, hi, frame in ((500, 550, blue), (550, 600, red), (600, 650, red))
    )
    path = make_path(n=24)
    observatory = with_instrument(
        replace(
            path,
            telescope=replace(
                path.telescope,
                pupil=SimplePupil(
                    circumscribed_diameter_m=6.0,
                    inscribed_diameter_m=4.0,
                    collecting_area_m2=20.0,
                ),
            ),
        ),
        coronagraph=ox.MultiBandCoronagraph(children),
    )
    first = Band(530.0, 560.0)
    for bands, waves in (
        ((first,), [[555.0]]),
        ((first, Band(610.0, 630.0)), [[555.0], [620.0]]),
    ):
        q = Quadrature(bands, jnp.array(waves), jnp.ones((len(bands), 1)))
        response = cp.aperture_response(
            _sel(observatory),
            q,
            rule=_RULE,
            n_separations=2,
            n_angles=1,
            max_separation_arcsec=0.03,
        )
        radius = _RULE.radius_lod * float(blue.lod_to_arcsec(545.0))
        radius /= float(_sel(observatory).detector.pixel_scale_arcsec)
        np.testing.assert_array_equal(
            response.radius_pix[0], np.asarray(radius, dtype=response.radius_pix.dtype)
        )
        np.testing.assert_array_equal(response.throughput_supported[0], True)


@pytest.mark.parametrize("product", ["band_average", "single_wavelength", "straddling"])
def test_aperture_reads_each_nodes_product_and_scale(gaussian_observation, product):
    """Both aperture moments use the analytic Gaussian on each detector pixel.

    The circle weights are inputs. Independent SciPy integrals, multiplied by
    the covering child's throughput, give the ring and probe fractions.
    The peak-scaled bound covers 128 operations for detector geometry,
    four erf endpoints, their differences and the aperture products,
    plus the weighted-sum reduction depth.
    """
    from conftest import IntegratedGaussianCoronagraph, gaussian_pixels

    from coronagraphoto.aperture import _weights

    coro = None
    if product == "straddling":
        blue = IntegratedGaussianCoronagraph(Band(480, 543), False, throughput=0.1)
        red = IntegratedGaussianCoronagraph(Band(543, 720), False, throughput=0.7)
        coro = ox.MultiBandCoronagraph((blue, red))
    _, observatory, _, sampling = gaussian_observation(
        product == "band_average", coro=coro
    )
    q = sampling.quadrature
    detector = _sel(observatory).detector
    frame = _sel(observatory).coronagraph.frame
    separation = 0.015
    response = cp.aperture_response(
        _sel(observatory),
        q,
        n_separations=2,
        n_angles=1,
        max_separation_arcsec=separation,
    )
    waves = np.asarray(q.nodes_nm[0])
    center = q.bands[0].center_nm
    radius = (
        _RULE.radius_lod
        * float(frame.lod_to_arcsec(center))
        / float(detector.pixel_scale_arcsec)
    )

    def moments(wave, x_arcsec, y_arcsec):
        scale_nm = center if product == "band_average" else wave
        lod = float(frame.lod_to_arcsec(scale_nm))
        pixels = gaussian_pixels(
            detector, frame, scale_nm, x_arcsec / lod, y_arcsec / lod, 0.35
        )
        amplitude = (0.1 if wave < 543 else 0.7) if coro is not None else 0.3
        pitch = float(detector.pixel_scale_arcsec)
        weights = _weights(
            _RULE, frame, detector.shape, radius, x_arcsec / pitch, y_arcsec / pitch
        )
        return [amplitude * np.sum(weights**power * pixels) for power in (1, 2)]

    dtype = response.throughput.dtype
    rtol = roundoff(1, n_ops=128, dtype=dtype) + sum_roundoff(
        1.0, np.prod(detector.shape), dtype=dtype
    )
    angle = np.pi * (np.sqrt(5.0) - 1)
    assert np.all(response.throughput_supported)
    for k, (x, y) in enumerate(
        ((0, 0), (separation * np.cos(angle), separation * np.sin(angle)))
    ):
        truth = np.asarray([moments(w, x, y) for w in waves]).T
        np.testing.assert_allclose(
            response.throughput[:, 0, :, k], truth, atol=rtol * truth.max(), rtol=0
        )
    probe = moments(waves[len(waves) // 2], separation, 0)[0]
    np.testing.assert_allclose(response.probe[0], probe, rtol=0, atol=rtol * abs(probe))


# Admission, stamps, reads and the record clock.

_RESPONSE_OPTIONS = dict(n_separations=5, n_angles=4, max_separation_arcsec=0.04)
_ROOTS = dict.fromkeys(("scene", "observatory", "exposure", "sampling"))


def _with_second_configuration(observatory):
    """The observatory with a copy of its configuration named ``other``."""
    (imager,) = observatory.instruments
    return replace(observatory, instruments=(imager, replace(imager, name="other")))


@pytest.mark.parametrize("staged", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("fact", "label"),
    [
        ("configuration", "instrument"),
        ("layout", "layout"),
        pytest.param(
            "precision",
            "precision",
            marks=pytest.mark.float64_only(reason="narrows the hardware to float32"),
        ),
    ],
)
def test_a_response_for_other_hardware_is_refused_by_its_stamps(
    plateau, fact, label, staged
):
    """Fails if a response built for other hardware or nodes is admitted.

    Basis: exact. The response is built for another configuration of the same
    telescope, for another quadrature layout, or at another precision, and each
    refusal names its fact. In the staged call the response is a ``jax.jit``
    argument, so only its static stamps can decide, and the refusal still fires.
    """
    scene, observatory, exposure, sampling, response, _ = plateau
    target = observatory
    if fact == "configuration":
        target = _with_second_configuration(observatory)
        response = cp.aperture_response(
            target.select("other"), sampling.quadrature, **_RESPONSE_OPTIONS
        )
    elif fact == "layout":
        _, finer = _build(scene, observatory, _BANDS, field_lod=0.0, order=3)
        response = cp.aperture_response(
            _sel(observatory), finer.quadrature, **_RESPONSE_OPTIONS
        )
    else:
        target = canonical(observatory, precision="float32")

    def build(response):
        return cp.aperture_record(scene, target, exposure, response, sampling=sampling)

    with pytest.raises(StampMismatchError, match=label):
        (jax.jit(build) if staged else build)(response)


@pytest.mark.parametrize("change", ["node", "probe"])
def test_value_mismatches_are_refused_on_the_host_and_unverified_when_staged(
    plateau, change
):
    """Fails if a moved node or a changed probe table is admitted on the host.

    Basis: exact. The stamps of both responses equal the good one's, so only
    the table comparison can tell. Concrete inputs refuse. With the response a
    ``jax.jit`` argument the comparison cannot run: the call is admitted and
    ``check_response`` reports ``values`` among the unverified facts. Called
    on the host first, ``check_response`` refuses.
    """
    scene, observatory, exposure, sampling, response, _ = plateau
    if change == "node":
        bad = eqx.tree_at(
            lambda r: r.quadrature.nodes_nm,
            response,
            response.quadrature.nodes_nm + 0.5,
        )
        match = "other wavelength nodes"
    else:
        bad = eqx.tree_at(lambda r: r.probe, response, response.probe * 1.01)
        match = "recorded"
    assert bad.built_at == response.built_at
    selected, quadrature = _sel(observatory), sampling.quadrature
    with pytest.raises(ValueError, match=match):
        cp.aperture_record(scene, observatory, exposure, bad, sampling=sampling)
    with pytest.raises(ValueError, match=match):
        cp.check_response(bad, selected, quadrature)
    seen = []

    def staged(response):
        seen.append(cp.check_response(response, selected, quadrature))
        return cp.aperture_record(
            scene, observatory, exposure, response, sampling=sampling
        )

    jax.jit(staged)(bad)
    assert "values" in seen[0]
    assert "values" not in cp.check_response(response, selected, quadrature)


def test_unverified_products_are_admitted_and_a_changed_value_is_caught(plateau):
    """Fails if a hand-built response or record is refused, or a bad value passes.

    Basis: exact. Without stamps nothing can be compared by stamp, so the
    response is admitted on the table comparison alone, and a hand-built record
    (no stamps, no reads) reads like the built one.
    """
    scene, observatory, exposure, sampling, response, record = plateau
    hand = replace(response, built_at=(), reads=())
    assert cp.check_response(hand, _sel(observatory), sampling.quadrature)
    built = cp.aperture_record(scene, observatory, exposure, hand, sampling=sampling)
    assert built.planet_response.shape == record.planet_response.shape
    bare = replace(record, built_at=(), reads=())
    east, north = _offsets(0.03, 40.0)
    np.testing.assert_array_equal(
        cp.aperture_rates(bare, east, north, contrast=1e-10, time_d=0.0).star_leakage,
        cp.aperture_rates(record, east, north, contrast=1e-10, time_d=0.0).star_leakage,
    )
    bad = eqx.tree_at(lambda r: r.probe, hand, hand.probe * 1.01)
    with pytest.raises(ValueError, match="recorded"):
        cp.aperture_record(scene, observatory, exposure, bad, sampling=sampling)


def test_reads_are_owner_qualified_and_stale_where_the_model_is_edited(sloped):
    """Fails if a read is scene-relative, drops the nodes, or a live edit is unseen.

    Basis: exact. Every read starts with its owner. Editing the coronagraph's
    sigma after the response was built is named by ``stale_reads`` against the
    response's reads, moves the image path and leaves the record's tables, so
    the old tables are the old answer bit for bit. Freeing each path the
    record reads is named. The quadrature nodes are among the reads of both.
    """
    scene, observatory, exposure, sampling, response, record = sloped
    assert "sampling.quadrature.nodes_nm" in response.reads
    assert "sampling.quadrature.nodes_nm" in record.reads
    assert all(r.split(".")[0] in _ROOTS for r in record.reads)
    write = "instruments[imager].coronagraph.sigma_lod"
    roots = {**_ROOTS, "observatory": ""}
    stale = stale_reads(bind_reads(response.reads, roots), (write,))
    assert stale and stale[0][0] == write
    wider = copy.copy(_sel(observatory).coronagraph)
    object.__setattr__(wider, "sigma_lod", 0.5)
    edited = with_instrument(observatory, coronagraph=wider)
    east, north = _offsets(0.05, 60.0)
    old = cp.aperture_rates(record, east, north, contrast=1e-10, time_d=0.0)
    before = cp.expected(scene, observatory, exposure, sampling=sampling)
    after = cp.expected(scene, edited, exposure, sampling=sampling)
    assert not np.array_equal(before.total(), after.total())
    new = cp.aperture_rates(record, east, north, contrast=1e-10, time_d=0.0)
    for one, two in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
        np.testing.assert_array_equal(one, two)
    for path in record.reads:
        owner, _, rest = path.partition(".")
        bound = bind_reads(record.reads, {**_ROOTS, owner: ""})
        assert stale_reads(bound, (rest,)), path


@pytest.mark.medium(reason="one record on a pointing-dependent zodi")
def test_read_time_inputs_move_the_rates_of_one_record(sloped):
    """Fails if a read-time contrast, position or date is baked into the record.

    Basis: exact. On one record, a different contrast moves the planet, a
    different position moves the leakage, and a different date moves the local
    zodi that follows the pointing.
    """
    scene, observatory, exposure, sampling, response, _ = sloped
    zodi = with_zodi(scene, LeinertZodi())
    record = cp.aperture_record(
        zodi, observatory, exposure, response, sampling=sampling
    )
    east, north = _offsets(0.04, 60.0)

    def read(**kwargs):
        options = dict(east_arcsec=east, north_arcsec=north, contrast=1e-10, time_d=0.0)
        return cp.aperture_rates(record, **{**options, **kwargs})

    base = read()
    assert not np.array_equal(read(contrast=2e-10).planet, base.planet)
    assert not np.array_equal(read(east_arcsec=-east).star_leakage, base.star_leakage)
    assert not np.array_equal(read(time_d=90.0).local_zodi, base.local_zodi)


def test_the_record_clock_counts_from_the_scene_epoch(plateau, precision):
    """Fails if the build date is not stated on the scene's day zero.

    Basis: algebra. The event at E + 12 d on a scene with epoch E gives 12, and
    the same instant on a scene with epoch E + 10 d gives 2, with equal
    backgrounds. A scene of another epoch is refused by the read. A float32 date
    cannot be shifted to another epoch, and the build says so.
    """
    scene, observatory, _, _, response, _ = plateau
    epoch = scene.epoch
    later = Epoch(epoch.jd1 + 10.0, epoch.jd2, epoch.scale)
    records = []
    other = make_system(separation_arcsec=0.03, epoch=later)
    if precision == np.dtype("float32"):
        exposure = cp.build_exposure(
            _BANDS, epoch=epoch, instrument="imager", time_d=12.0
        )
        with pytest.raises(PrecisionError, match="offset"):
            cp.build_sampling(observatory, exposure, scene=other, field_lod=0.0)
            cp.aperture_record(
                other,
                observatory,
                exposure,
                response,
                sampling=cp.build_sampling(
                    observatory, exposure, scene=scene, field_lod=0.0
                ),
            )
        return
    for target in (scene, other):
        exposure = cp.build_exposure(
            _BANDS, epoch=epoch, instrument="imager", time_d=12.0
        )
        sampling = cp.build_sampling(observatory, exposure, scene=target, field_lod=0.0)
        records.append(
            cp.aperture_record(
                target, observatory, exposure, response, sampling=sampling
            )
        )
    first, second = records
    np.testing.assert_allclose(
        [first.build_time_d, second.build_time_d],
        [12.0, 2.0],
        rtol=float(ulps(4, np.float64(1))),
        atol=0,
    )
    assert (first.epoch, second.epoch) == (epoch, later)
    np.testing.assert_array_equal(first.star_leakage, second.star_leakage)
    with pytest.raises(ValueError, match="epoch"):
        cp.scene_aperture_rates(second, scene, jnp.array([0.0]))


def test_records_refuse_to_stack_across_epochs_and_stamps(sloped):
    """Fails if records of other epochs or hardware stack, or the field is unnamed.

    Basis: exact tree structure. The stamps hold the layout and not node
    values, so a response built on moved nodes of one layout has equal stamps.
    """
    _, observatory, _, sampling, response, record = sloped
    stack = lambda *leaves: jnp.stack(leaves)  # noqa: E731
    jax.tree.map(stack, record, record)
    shifted = eqx.tree_at(
        lambda q: q.nodes_nm, sampling.quadrature, sampling.quadrature.nodes_nm + 1.0
    )
    moved = cp.aperture_response(
        _sel(observatory), shifted, n_separations=9, n_angles=8
    )
    assert moved.built_at == response.built_at
    epoch = Epoch(record.epoch.jd1 + 1.0, record.epoch.jd2, record.epoch.scale)
    # JAX reports the differing static metadata (the value, not the field name).
    for field, value, shown in (
        ("epoch", epoch, "jd1=2451546.0"),
        ("built_at", (("producer", "str:another"),), "str:another"),
    ):
        other = dataclasses.replace(record, **{field: value})
        with pytest.raises(ValueError, match=shown):
            jax.tree.map(stack, record, other)


def test_record_gain_includes_the_telescope_train(plateau):
    """Fails if the collecting gain omits an element shared by every configuration.

    Basis: exact algebra. A constant 0.5 on the telescope's own train halves
    every planet table of the record, to four roundings of the gain product.
    """
    scene, observatory, exposure, sampling, response, record = plateau
    shared = replace(
        observatory,
        telescope=replace(
            observatory.telescope, attenuating_elements=(ConstantThroughput(0.5),)
        ),
    )
    halved = cp.aperture_record(scene, shared, exposure, response, sampling=sampling)
    np.testing.assert_allclose(
        halved.planet_response,
        0.5 * record.planet_response,
        rtol=float(ulps(4, np.float64(1))),
        atol=0,
    )


def test_default_rule_and_keyword_offsets(plateau):
    """Fails if the rule default drifts or the offsets stop reading by keyword.

    Basis: exact. Omitting ``rule`` builds the released default rule, and the
    offsets by keyword equal the positional call.
    """
    _, observatory, _, sampling, _, record = plateau
    explicit = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        rule=DEFAULT_APERTURE_RULE,
        **_RESPONSE_OPTIONS,
    )
    default = cp.aperture_response(
        _sel(observatory), sampling.quadrature, **_RESPONSE_OPTIONS
    )
    np.testing.assert_array_equal(default.kernels, explicit.kernels)
    assert dict(default.built_at)["rule"] == dict(explicit.built_at)["rule"]
    east, north = _offsets(0.03, 40.0)
    by_keyword = cp.aperture_rates(
        record, east_arcsec=east, north_arcsec=north, contrast=1e-10, time_d=0.0
    )
    by_position = cp.aperture_rates(record, east, north, contrast=1e-10, time_d=0.0)
    for one, two in zip(
        jax.tree.leaves(by_keyword), jax.tree.leaves(by_position), strict=True
    ):
        np.testing.assert_array_equal(one, two)


def test_aperture_stamps_carry_the_producer_version(plateau):
    """Fails if a built aperture product does not say which release built it."""
    *_, response, record = plateau
    assert dict(response.built_at)["producer_version"] == f"str:{__version__}"
    assert dict(record.built_at)["producer_version"] == f"str:{__version__}"
    assert dict(record.built_at)["response.producer_version"] == f"str:{__version__}"


def test_a_multiband_product_reads_every_routed_child_and_no_other():
    """Fails if a routed child's edit is unseen, or an unrouted child is named.

    Basis: exact. Bands centred at 525 and 575 nm route to the first two of
    three children, so editing the second is named by ``stale_reads`` with its
    own path, and the third, which no band reaches, is absent.
    """
    frame = Frame(550.0, 6.0, "circumscribed")
    children = tuple(
        GaussianCoronagraph(n=24, bandpass=Band(lo, hi), frame=frame)
        for lo, hi in ((500, 550), (550, 600), (600, 650))
    )
    observatory = with_instrument(
        make_path(n=24), coronagraph=ox.MultiBandCoronagraph(children)
    )
    quadrature = Quadrature(
        (Band(510.0, 540.0), Band(560.0, 590.0)),
        jnp.array([[525.0], [575.0]]),
        jnp.ones((2, 1)),
    )
    response = cp.aperture_response(
        _sel(observatory),
        quadrature,
        n_separations=2,
        n_angles=1,
        max_separation_arcsec=0.03,
    )
    roots = {**_ROOTS, "observatory": ""}
    bound = bind_reads(response.reads, roots)
    base = "instruments[imager].coronagraph.coronagraphs"
    assert stale_reads(bound, (f"{base}[1].sigma_lod",))
    assert stale_reads(bound, (f"{base}[0].sigma_lod",))
    assert not stale_reads(bound, (f"{base}[2].sigma_lod",))
    assert not any("coronagraphs[2]" in read for read in response.reads)


@pytest.mark.medium(reason="a path coronagraph calibration and one response")
def test_a_path_coronagraph_response_carries_calibration_and_refuses_when_stale():
    """Fails if a stale calibration is read through a cached reader.

    Basis: exact. The response carries the calibration stamps of the path model.
    After the stop is edited without ``recalibrated()`` (by a factor that leaves
    the live probe equal), a record builder that reads the cached sky map
    raises, and the response built from the live readers alone still runs.
    """
    pytest.importorskip("physicaloptix")
    import importlib

    interop = importlib.import_module("physicaloptix.interop")
    core = importlib.import_module("physicaloptix.core")
    elements = importlib.import_module("physicaloptix.elements")
    path_module = importlib.import_module("physicaloptix.path")
    grid = ox.grids.PixelGrid.pupil(16)
    stop = elements.SampledOptic(jnp.ones(grid.shape), grid, core.PlaneKind.PUPIL)
    pupil = SimplePupil(
        circumscribed_diameter_m=6.0, collecting_area_m2=np.pi * 6.0**2 / 4
    )
    coronagraph = interop.PathCoronagraph.from_path(
        path_module.OpticalPath((path_module.Stage("stop", stop),)),
        core.Field(jnp.ones(grid.shape, dtype=complex), grid, core.PlaneKind.PUPIL),
        pupil=pupil,
        reference_wavelength_nm=600.0,
        max_separation_lod=2.5,
        pixel_scale_lod=0.25,
        n_sky_screens=4,
        seed=17,
    )
    scale = float(pupil.frame(600.0).lod_to_arcsec(600.0))
    detector = Detector(
        pixel_scale_arcsec=0.25 * scale,
        shape=(8, 8),
        qe_curve=([500, 700], [0.9, 0.9]),
    )
    observatory = observatory_for(pupil, (), coronagraph, detector)
    scene = make_system(separation_arcsec=0.5 * scale, smooth_planet=True)
    exposure, sampling = _build(scene, observatory, (Band(590.0, 610.0),), order=2)
    response = cp.aperture_response(
        _sel(observatory),
        sampling.quadrature,
        n_separations=2,
        n_angles=1,
        max_separation_arcsec=0.5 * scale,
    )
    assert any(label.startswith("coronagraph.") for label, _ in response.built_at)
    cp.aperture_record(scene, observatory, exposure, response, sampling=sampling)
    edited = eqx.tree_at(
        lambda o: o.instruments[0].coronagraph.core_path.stages[-1].op.transmission,
        observatory,
        replace_fn=lambda t: (1.0 + 1e-6) * t,
    )
    with pytest.raises(StampMismatchError, match="recalibrated"):
        cp.aperture_record(scene, edited, exposure, response, sampling=sampling)
    live = cp.aperture_response(
        _sel(edited),
        sampling.quadrature,
        n_separations=2,
        n_angles=1,
        max_separation_arcsec=0.5 * scale,
    )
    assert live.kernels.shape == response.kernels.shape
