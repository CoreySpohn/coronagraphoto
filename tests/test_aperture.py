"""Aperture response, record and read against closed forms and image sums."""

import copy

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
    with_zodi,
    write_yip,
)
from hwoutils.constants import Rsun2m
from optixstuff import (
    ConstantThroughput,
    Detector,
    FramePlan,
    Instrument,
    SimplePrimary,
)
from optixstuff.apertures import Circular, HalfMaximum
from optixstuff.grids import Image, Measure, SkyOrientation
from optixstuff.spectral import Band
from skyscapes import AYOZodi, LeinertZodi
from skyscapes.physical_model import BlackbodyPhysicalModel, GridPhysicalModel
from skyscapes.scene import BlackbodyStar, Planet
from yippy import open_yip

import coronagraphoto as cp

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
    """Worked instrument with 0.3 lambda/D pixels and the given coronagraph."""
    path = make_path(detector_n=48, detector_step=0.3)
    return eqx.tree_at(lambda p: p.coronagraph, path, coronagraph)


def _offsets(separation, position_angle_deg):
    """East and north offsets of a planet at a separation and position angle."""
    angle = jnp.deg2rad(position_angle_deg)
    return separation * jnp.sin(angle), separation * jnp.cos(angle)


def _image_weights(instrument, x_arcsec, y_arcsec):
    """The rule's pixel weights for an aperture at a detector position."""
    grid = instrument.detector.grid
    frame = instrument.coronagraph.grid.frame
    scale = float(frame.lod_to_arcsec(550.0))
    ones = Image(
        jnp.ones(grid.shape),
        grid.to_lod(550.0, frame),
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    return _RULE.weights(ones, (float(x_arcsec) / scale, float(y_arcsec) / scale))


@pytest.fixture(scope="module")
def plateau():
    """Flat-PSF instrument, its response and one target's record."""
    instrument = _instrument(PlateauCoronagraph(40, 0.25))
    scene = make_system(separation_arcsec=0.03)
    # The record tables are compared with the image at the same four nodes, so
    # the layout is not under test here and the field is zero.
    exposure = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0, field_lod=0.0)
    response = cp.aperture_response(
        instrument,
        exposure.quadrature,
        n_separations=5,
        n_angles=4,
        max_separation_arcsec=0.04,
    )
    record = cp.aperture_record(scene, instrument, exposure, response)
    return scene, instrument, exposure, response, record


@pytest.fixture(scope="module")
def sloped():
    """Sloped-leakage instrument with a record rendered at zero roll."""
    instrument = _instrument(SlopedLeakage(40, 0.25))
    scene = make_system(separation_arcsec=0.05)
    # Table and image are compared at the same four nodes; see ``plateau``.
    exposure = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0, field_lod=0.0)
    response = cp.aperture_response(
        instrument, exposure.quadrature, n_separations=9, n_angles=8
    )
    record = cp.aperture_record(scene, instrument, exposure, response)
    return scene, instrument, exposure, response, record


@vv.case("aperture-photon-budget", "code-verification")
def test_worked_example_closed_form(plateau):
    """Fails if a radiometric factor or the fixed-angle aperture area is wrong.

    The manufactured PSF is flat, 0.3/16 per square lambda/D, and the leakage
    is 1e-9 per square lambda/D, so every aperture sum is an area: the circle
    of 0.7 lambda/D at the 550 nm band centre covers pi (0.7 * 550 / lambda)**2
    square lambda/D at node lambda, 19% apart between the outer nodes.
    Basis: exact areas for the rates, within 3e-2 relative: the sharp edges of
    the manufactured fields ring under band-limited resampling onto 0.3
    lambda/D pixels, measured at up to 2.1e-2. The pixel count is exact.
    """
    _, _, exposure, response, record = plateau
    nodes = np.asarray(exposure.quadrature.nodes_nm)
    area = np.pi * (0.7 * 550.0 / nodes) ** 2
    rates = cp.aperture_rates(record, *_offsets(0.03, 40.0), 1e-10, time_d=0.0)
    incident = 1e6 * 40 * 0.4
    np.testing.assert_allclose(response.radius_pix, [0.7 / 0.3], rtol=1e-12)
    np.testing.assert_allclose(
        rates.planet, 1e-10 * incident * 0.3 / 16 * area, rtol=3e-2
    )
    np.testing.assert_allclose(rates.star_leakage, incident * 1e-9 * area, rtol=3e-2)
    n_pix = np.pi * (0.7 / 0.3) ** 2
    np.testing.assert_allclose(rates.n_pix, [n_pix], rtol=1e-12)
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
    both sides; 1e-9 relative allows float64 rounding of the convolution.
    """
    scene, instrument, exposure, _, record = sloped
    photons = cp.photon_rates(scene, instrument, exposure)
    pitch = float(instrument.detector.pixel_scale_arcsec)
    roll = 25.0
    for dx, dy in ((3.0, -2.5), (-4.5, 1.0), (0.5, 6.5)):
        x, y = dx * pitch, dy * pitch
        east, north = SkyOrientation(roll).detector_to_sky(x, y)
        rates = cp.aperture_rates(
            record,
            east,
            north,
            1e-10,
            time_d=0.0,
            roll_deg=roll,
        )
        weights = _image_weights(instrument, x, y)
        for moment, power in ((rates, 1), (rates.variance, 2)):
            np.testing.assert_allclose(
                moment.star_leakage,
                jnp.einsum("wqyx,yx->wq", photons.star_leakage, weights**power),
                rtol=1e-9,
            )
            np.testing.assert_allclose(
                moment.n_pix, [float((weights**power).sum())], rtol=1e-9
            )
    # The slope is seen: two positions differ by far more than the tolerance.
    first, second = (
        cp.aperture_rates(record, *_offsets(0.05, pa), 1e-10, time_d=0.0).star_leakage
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
    neighbouring knots.
    """
    scene, instrument, exposure, response, record = sloped
    photons = cp.photon_rates(scene, instrument, exposure)
    east, north = scene.positions(jnp.array([0.0]))[:2, 0, 0]
    separation = float(jnp.hypot(east, north))
    x, y = SkyOrientation(0.0).sky_to_detector(east, north)
    weights = _image_weights(instrument, x, y)
    rates = cp.aperture_rates(
        record,
        east,
        north,
        1e-10,
        time_d=0.0,
    )
    knot = int(separation / float(response.separation_step_arcsec))
    table = np.asarray(response.throughput[0, 0])
    scatter = np.asarray(response.throughput_ring_std[0])[:, knot : knot + 2].max(-1)
    curvature = abs(np.diff(table[:, knot - 1 : knot + 3], n=2, axis=-1)).max(-1) / 8
    tolerance = (3 * scatter + curvature) / table[:, knot : knot + 2].min(-1)
    truth = jnp.einsum("wqyx,yx->wq", photons.planet, weights)
    assert np.all(tolerance < 0.05)
    assert np.all(abs(np.asarray(rates.planet / truth)[0] - 1) <= tolerance)
    # Contrast factors out of the planet and touches nothing else.
    doubled = cp.aperture_rates(record, *_offsets(separation, 90.0), 2e-10, time_d=0.0)
    single = cp.aperture_rates(record, *_offsets(separation, 90.0), 1e-10, time_d=0.0)
    np.testing.assert_allclose(doubled.planet, 2 * single.planet, rtol=1e-14)
    np.testing.assert_array_equal(doubled.star_leakage, single.star_leakage)
    per_node = cp.aperture_rates(
        record, *_offsets(separation, 90.0), jnp.full((1, 4), 1e-10), time_d=0.0
    )
    np.testing.assert_allclose(per_node.planet, single.planet, rtol=1e-14)


@vv.case("aperture-photon-budget", "code-verification")
def test_read_geometry_on_a_hand_made_record(plateau):
    """Fails if position angle, roll, parity or the sky-fixed exozodi is wrong.

    The tables are set by hand to each lattice position's pixel coordinate,
    which bilinear interpolation reproduces exactly, so every read returns
    where it looked. The expected detector position is written out for
    detector +y at position angle ``roll`` and east to the left at zero roll:
    x = -(E cos r - N sin r), y = E sin r + N cos r. Basis: exact, 1e-12.
    """
    record = plateau[4]
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
    pitch, centre = float(record.pixel_scale_arcsec), (shape[0] / 2 - 1) / 2

    def expected(separation, angle, roll):
        e = separation * np.sin(np.deg2rad(angle))
        n = separation * np.cos(np.deg2rad(angle))
        c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
        return centre - (e * c - n * s) / pitch, centre + (e * s + n * c) / pitch

    separation, angle = 0.031, 70.0
    built = float(record.roll_deg)
    rates = cp.aperture_rates(record, *_offsets(separation, angle), 1e-10, time_d=0.0)
    column, row = expected(separation, angle, built)
    np.testing.assert_allclose(rates.star_leakage, column, rtol=1e-12)
    np.testing.assert_allclose(rates.local_zodi, row, rtol=1e-12)
    np.testing.assert_allclose(rates.exozodi, column, rtol=1e-12)
    # East is to the left and north is up at zero roll.
    assert built == 0.0 and column < centre < row
    rolled = cp.aperture_rates(
        record, *_offsets(separation, angle), 1e-10, time_d=0.0, roll_deg=115.0
    )
    new_column, new_row = expected(separation, angle, 115.0)
    assert abs(new_column - column) > 1 and abs(new_row - row) > 1
    np.testing.assert_allclose(rolled.star_leakage, new_column, rtol=1e-12)
    np.testing.assert_allclose(rolled.local_zodi, new_row, rtol=1e-12)
    # The exozodi stays with the sky position it was rendered at.
    np.testing.assert_allclose(rolled.exozodi, column, rtol=1e-12)
    np.testing.assert_allclose(rolled.planet, rates.planet, rtol=1e-14)


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
    rounding of the aperture sum, measured 4e-15 relative and held to 1e-12.
    The rebuilt record holds the same uniform-sky table, so its read must be
    bit-identical in every term.
    """
    scene, instrument, _, response, _ = sloped
    scene = with_zodi(scene, LeinertZodi())
    scene = eqx.tree_at(lambda s: s.dynamics.frame.dec_deg, scene, jnp.asarray(23.0))
    exposures = [
        cp.build_exposure(scene, instrument, _BANDS, time_d=t, field_lod=0.0)
        for t in (0.0, 137.0)
    ]
    built, rebuilt = (
        cp.aperture_record(scene, instrument, exposure, response)
        for exposure in exposures
    )
    pitch = float(instrument.detector.pixel_scale_arcsec)
    x, y = 3.0 * pitch, -2.5 * pitch
    east, north = SkyOrientation(0.0).detector_to_sky(x, y)

    def read(record, time_d):
        return cp.aperture_rates(
            record,
            east,
            north,
            1e-10,
            time_d=time_d,
        )

    late, reference = read(built, 137.0), read(rebuilt, 137.0)
    photons = cp.photon_rates(scene, instrument, exposures[1])
    weights = _image_weights(instrument, x, y)
    for moment, expected, power in (
        (late, reference, 1),
        (late.variance, reference.variance, 2),
    ):
        assert np.all(moment.local_zodi > 0)
        np.testing.assert_allclose(
            moment.local_zodi,
            jnp.einsum("wqyx,yx->wq", photons.local_zodi, weights**power),
            rtol=1e-12,
        )
        for name in ("planet", "star_leakage", "local_zodi", "exozodi", "n_pix"):
            np.testing.assert_array_equal(
                getattr(moment, name), getattr(expected, name)
            )
    # The time is seen, and by exactly the radiance ratio of the sightlines.
    nodes = exposures[0].quadrature.nodes_nm
    radiance = [leinert_radiance(scene, nodes, t) for t in (0.0, 137.0)]
    early = read(built, 0.0)
    assert np.all(abs(radiance[1] / radiance[0] - 1) > 0.5)
    np.testing.assert_allclose(
        late.local_zodi / early.local_zodi, radiance[1] / radiance[0], rtol=1e-13
    )
    # One record serves a batch of times under one compiled program.
    batched = jax.jit(jax.vmap(lambda t: read(built, t).local_zodi))(
        jnp.array([0.0, 137.0])
    )
    np.testing.assert_allclose(
        batched, jnp.stack([early.local_zodi, late.local_zodi]), rtol=1e-13
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_read_under_jax_transformations(sloped):
    """Fails if a batched, compiled or differentiated read departs from a plain one.

    Basis: vmap and jit equal eager reads to a 1e-13 relative rounding
    allowance. The read is piecewise bilinear, so within one lattice cell a
    central difference of step 1e-7 arcsec has no truncation error and the
    1e-6 relative tolerance covers its rounding.
    """
    record, detector = sloped[4], sloped[1].detector

    def read(separation, angle, contrast):
        rates = cp.aperture_rates(
            record, *_offsets(separation, angle), contrast, time_d=0.0
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
    np.testing.assert_allclose(batched, looped, rtol=1e-13)

    step = 1e-7
    slope = jax.grad(read)(0.04, 200.0, 3e-10)
    difference = (read(0.04 + step, 200.0, 3e-10) - read(0.04 - step, 200.0, 3e-10)) / (
        2 * step
    )
    assert slope != 0
    np.testing.assert_allclose(slope, difference, rtol=1e-6)
    by_contrast = jax.grad(read, argnums=2)(0.04, 200.0, 3e-10)
    difference = (read(0.04, 200.0, 3.03e-10) - read(0.04, 200.0, 2.97e-10)) / 6e-12
    np.testing.assert_allclose(by_contrast, difference, rtol=1e-4)


@vv.case("aperture-photon-budget", "code-verification")
def test_reads_outside_the_tables_and_unsupported_builds(sloped):
    """Fails if a read outside the tables returns light or a bad build passes.

    Basis: exact. An unsupported read is zero in every field with ``supported``
    False, so ``snr`` is NaN and ``valid`` clears downstream, and a rule or
    quadrature the tables cannot represent raises at build time.
    """
    scene, instrument, exposure, response, record = sloped
    detector = instrument.detector
    last = float(response.separation_step_arcsec) * 8
    inside = cp.aperture_rates(record, *_offsets(0.5 * last, 0.0), 1e-10, time_d=0.0)
    assert np.all(np.isfinite(inside.planet)) and np.all(np.isfinite(inside.n_pix))
    assert np.all(inside.supported) and np.all(inside.planet > 0)
    for separation in (1.01 * last, np.nan):
        outside = cp.aperture_rates(
            record, *_offsets(separation, 0.0), 1e-10, time_d=0.0
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
    edge = cp.aperture_rates(record, *_offsets(half * 0.99, 90.0), 1e-10, time_d=0.0)
    assert not np.any(edge.supported)
    assert np.all(np.asarray(edge.star_leakage) == 0.0)
    assert np.all(np.asarray(edge.n_pix) == 0.0)

    for rule in (HalfMaximum(), Circular(0.7, centre="peak")):
        with pytest.raises(NotImplementedError, match="Circular"):
            cp.aperture_response(instrument, exposure.quadrature, rule=rule)
    other = cp.build_exposure(scene, instrument, (Band(510.0, 590.0),), time_d=0.0)
    with pytest.raises(ValueError, match="other wavelength nodes"):
        cp.aperture_record(scene, instrument, other, response)


def test_record_refuses_a_response_built_for_another_instrument(plateau):
    """Fails if a swapped coronagraph, pixel scale or band pairs with an old response.

    Basis: exact. Each swap keeps the wavelength nodes of the response and
    changes one thing: the coronagraph tables (a Gaussian for the flat PSF,
    seen only by the measured source), the detector pixel scale (seen in the
    aperture radius) and the band edges (seen in the bands). The instrument
    the response was built for passes.
    """
    scene, instrument, exposure, response, _ = plateau
    cp.aperture_record(scene, instrument, exposure, response)
    swapped = _instrument(GaussianCoronagraph(40, 0.25))
    with pytest.raises(ValueError, match="another coronagraph or detector"):
        cp.aperture_record(scene, swapped, exposure, response)
    detector = instrument.detector
    coarse = eqx.tree_at(
        lambda i: i.detector,
        instrument,
        Detector(
            pixel_scale_arcsec=1.1 * float(detector.pixel_scale_arcsec),
            shape=detector.shape,
        ),
    )
    with pytest.raises(ValueError, match="pixel scale"):
        cp.aperture_record(scene, coarse, exposure, response)
    shifted = copy.copy(response.quadrature)
    object.__setattr__(shifted, "bands", (Band(500.0, 600.0001),))
    other = eqx.tree_at(lambda r: r.quadrature, response, shifted)
    with pytest.raises(ValueError, match="other bands"):
        cp.aperture_record(scene, instrument, exposure, other)


def test_flux_scale_scales_every_record_table_and_passes_the_guard(plateau):
    """Fails if the flux scale misses a table or invalidates the response.

    Basis: exact algebra, 1e-6 relative float32 rounding allowance. The scale
    acts once, in the effective area, so every rate table of a record built
    with scale 1.5 is 1.5 times the unit-scale table from the same response,
    and the pixel counts are unchanged.
    """
    scene, instrument, exposure, response, record = plateau
    scaled = eqx.tree_at(lambda i: i.flux_scale, instrument, jnp.asarray(1.5))
    other = cp.aperture_record(scene, scaled, exposure, response)
    for name in ("planet_per_contrast", "star_leakage", "uniform_sky", "exozodi"):
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
    doubles only the planet, each by its factor within 1e-5 relative float32
    rounding; the record and the contrasts are rebuilt for each scene.
    """
    _, instrument, exposure, response, _ = plateau
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
        record = cp.aperture_record(scene, instrument, exposure, response)
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
    _, instrument, exposure, response, _ = plateau
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
    record = cp.aperture_record(scene, instrument, exposure, response)
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
            record, east[0, t], north[0, t], contrast, time_d=time
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
        instrument.detector,
        7.0,
        frame_time_s=100.0,
        post_processing=_NO_RESIDUAL,
    )
    assert solved.time_s.shape == solved.valid.shape == (1, 2, 3)
    assert np.all(np.asarray(solved.valid[:, 0])) and not solved.valid[:, 1].any()
    assert np.all(np.isnan(np.asarray(solved.time_s[:, 1])))
    with pytest.raises(ValueError, match="shape"):
        cp.scene_aperture_rates(record, scene, 0.0)


def _warm(instrument, temperature_K=290.0, emissivity=0.5):
    """The instrument with a lumped warm-telescope emission."""
    return eqx.tree_at(
        lambda p: p.warm_emission,
        instrument,
        ox.WarmEmission(temperature_K, emissivity),
        is_leaf=lambda x: x is None,
    )


def _pixel_thermal_rate(instrument, nodes_nm, temperature_K, emissivity):
    """Warm emission per detector pixel in photons/s/nm, from primitives.

    Planck photon radiance 2 c / lambda**4 / expm1(h c / (lambda k T)) per
    meter of wavelength, converted to per nanometer, times the emissivity, the
    geometric collecting area and the pixel solid angle.
    """
    from scipy import constants

    wavelength_m = np.asarray(nodes_nm) * 1e-9
    x = constants.h * constants.c / (wavelength_m * constants.k * temperature_K)
    radiance = 2 * constants.c / wavelength_m**4 / np.expm1(x) * 1e-9
    pixel_sr = np.deg2rad(instrument.detector.pixel_scale_arcsec / 3600.0) ** 2
    return emissivity * radiance * instrument.primary.collecting_area_m2 * pixel_sr


def test_thermal_record_and_scene_read(plateau):
    """Fails if the record drops, double counts or misshapes warm emission.

    Basis: the per-pixel rate from primitives times the aperture pixel count,
    for the mean and the squared-weight records, at rtol 1e-12 in float64
    (a Planck exponential and a four-factor product); the scalar read and the
    scene read equal exactly; an invalid planet reads zero; an instrument
    without warm emission reads exact zeros.
    """
    _, cold, exposure, response, _ = plateau
    instrument = _warm(cold)
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
    record = cp.aperture_record(scene, instrument, exposure, response)
    pixel = _pixel_thermal_rate(instrument, record.quadrature.nodes_nm, 290.0, 0.5)
    assert record.thermal.shape == (1, 4)
    np.testing.assert_allclose(record.thermal, pixel, rtol=1e-12)
    times = jnp.array([0.0, 30.0, 90.0])
    rates = cp.scene_aperture_rates(record, scene, times)
    assert rates.thermal.shape == (1, 4, 2, 3)
    assert rates.variance.thermal.shape == (1, 4, 2, 3)
    east, north = scene.positions(times)
    for t, time in enumerate(times):
        single = cp.aperture_rates(record, east[0, t], north[0, t], 1e-10, time_d=time)
        np.testing.assert_allclose(
            single.thermal, pixel * single.n_pix[:, None], rtol=1e-12
        )
        np.testing.assert_allclose(
            single.variance.thermal,
            pixel * single.variance.n_pix[:, None],
            rtol=1e-12,
        )
        np.testing.assert_array_equal(rates.thermal[..., 0, t], single.thermal)
        np.testing.assert_array_equal(
            rates.variance.thermal[..., 0, t], single.variance.thermal
        )
    assert np.all(np.asarray(rates.thermal[..., 1, :]) == 0.0)
    assert np.all(np.asarray(rates.thermal[..., 0, :]) > 0.0)
    off = cp.aperture_rates(record, 1.0, 1.0, 1e-10, time_d=0.0)
    assert not np.any(off.supported)
    np.testing.assert_array_equal(off.thermal, 0.0)
    cold_record = cp.aperture_record(scene, cold, exposure, response)
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
    scene, instrument, exposure, response, _ = sloped
    detector = Detector(
        shape=instrument.detector.shape,
        pixel_scale_arcsec=instrument.detector.pixel_scale_arcsec,
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=2e-3,
        clock_induced_charge_e_per_pixel_frame=0.01,
        read_noise_e_rms=0.3,
    )
    instrument = eqx.tree_at(lambda p: p.detector, instrument, detector)
    bright = eqx.tree_at(lambda s: s.star.flux_phot_per_nm_m2, scene, jnp.asarray(1e9))
    # The response holds no detector noise, so the fixture's one serves.
    record = cp.aperture_record(bright, instrument, exposure, response)
    east, north = bright.positions(jnp.array([0.0]))[:2, 0, 0]
    x, y = SkyOrientation(0.0).sky_to_detector(east, north)
    weights = np.asarray(_image_weights(instrument, x, y))
    rates = cp.aperture_rates(
        record,
        east,
        north,
        1e-7,
        time_d=0.0,
    )
    plan = FramePlan(50.0, 4)
    mean, variance = detector.mean_and_variance(detector.electrons(rates), plan)

    images = cp.expected(bright, instrument, exposure)
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
    _, instrument, _, response, _ = plateau
    scene = make_system(separation_arcsec=0.03)
    scene = eqx.tree_at(
        lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
        scene,
        (None, None),
        is_leaf=lambda x: x is None,
    )
    exposure = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0, field_lod=0.0)
    record = cp.aperture_record(
        with_zodi(scene, AYOZodi()), instrument, exposure, response
    )
    assert record.sky_frame.ra_deg is None
    rates = cp.aperture_rates(record, *_offsets(0.02, 0.0), 1e-10, time_d=5.0)
    assert float(jnp.sum(rates.local_zodi)) > 0
    pointed = cp.aperture_record(
        with_zodi(scene, LeinertZodi()), instrument, exposure, response
    )
    with pytest.raises(ValueError, match="no sky position"):
        cp.aperture_rates(pointed, *_offsets(0.02, 0.0), 1e-10, time_d=5.0)


@vv.case("aperture-photon-budget", "code-verification")
def test_records_of_two_targets_stack_and_read_under_vmap(sloped):
    """Fails if a record carries static per-target data a stack cannot hold.

    Basis: two targets of one shape differ only in sky position, so their
    records stack leaf by leaf and one ``vmap`` over the stack equals the two
    separate reads to a 1e-13 relative rounding allowance. The local zodi
    follows the pointing, so the two reads must also differ.
    """
    scene, instrument, exposure, response, _ = sloped
    scene = with_zodi(scene, LeinertZodi())
    other = eqx.tree_at(
        lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
        scene,
        (jnp.asarray(250.0), jnp.asarray(35.0)),
    )
    records = [
        cp.aperture_record(target, instrument, exposure, response)
        for target in (scene, other)
    ]
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *records)
    east, north = _offsets(0.04, 60.0)

    def read(record):
        return cp.aperture_rates(record, east, north, 1e-10, time_d=60.0)

    together = eqx.filter_vmap(read)(stacked)
    for i, record in enumerate(records):
        single = read(record)
        for name in ("planet", "star_leakage", "local_zodi", "exozodi", "n_pix"):
            np.testing.assert_allclose(
                getattr(together, name)[i], getattr(single, name), rtol=1e-13
            )
    zodi = np.asarray(together.local_zodi)
    assert np.all(abs(zodi[0] / zodi[1] - 1) > 0.01)


def test_record_builds_under_jit_and_jit_vmap_with_the_instrument_closed_over(sloped):
    """Fails if the response guard breaks a jitted record build.

    Basis: exact. The guard measures concrete values at trace time even
    inside ``jax.jit``, so ``aperture_record`` with the instrument closed
    over runs under ``jit`` and ``jit(vmap)`` over the scene's pointing, and
    equals the eager record to a 1e-13 relative rounding allowance.
    """
    scene, instrument, exposure, response, _ = sloped
    scene = with_zodi(scene, LeinertZodi())

    def build(ra_deg, dec_deg):
        pointed = eqx.tree_at(
            lambda s: (s.dynamics.frame.ra_deg, s.dynamics.frame.dec_deg),
            scene,
            (ra_deg, dec_deg),
        )
        return cp.aperture_record(pointed, instrument, exposure, response)

    ra, dec = jnp.asarray([250.0, 260.0]), jnp.asarray([35.0, 40.0])
    eager = [build(ra[i], dec[i]) for i in range(2)]
    jitted = jax.jit(build)(ra[0], dec[0])
    batched = jax.jit(jax.vmap(build))(ra, dec)
    for name in ("planet_per_contrast", "star_leakage", "exozodi", "n_pix"):
        np.testing.assert_allclose(
            getattr(jitted, name), getattr(eager[0], name), rtol=1e-13
        )
        for i in range(2):
            np.testing.assert_allclose(
                getattr(batched, name)[i], getattr(eager[i], name), rtol=1e-13
            )


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
    instrument = Instrument(
        primary=SimplePrimary(
            circumscribed_diameter_m=coro.frame.diameter_m, collecting_area_m2=10.0
        ),
        attenuating_elements=(ConstantThroughput(throughput=1.0),),
        coronagraph=coro,
        detector=Detector(pixel_scale_arcsec=0.1 * scale, shape=(96, 96), qe=1.0),
    )
    quadrature = cp.build_exposure(
        make_system(), instrument, (Band(549.0, 551.0),), time_d=0.0
    ).quadrature
    response = cp.aperture_response(
        instrument,
        quadrature,
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

    instrument = _instrument(GaussianCoronagraph(24, 0.25))
    instrument = eqx.tree_at(
        lambda p: p.detector,
        instrument,
        Detector(
            pixel_scale_arcsec=0.25
            * float(instrument.coronagraph.grid.frame.lod_to_arcsec(550.0)),
            shape=(48, 48),
            qe=1.0,
        ),
    )
    scale = float(instrument.coronagraph.grid.frame.lod_to_arcsec(550.0))
    scene = make_system(separation_arcsec=1.0 * scale)
    exposure = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0)
    response = cp.aperture_response(
        instrument, exposure.quadrature, n_separations=21, n_angles=8
    )
    record = cp.aperture_record(scene, instrument, exposure, response)

    def read(separation_lod):
        return cp.aperture_rates(record, separation_lod * scale, 0.0, 1e-10, time_d=0.0)

    near = read(1.0)
    assert bool(near.supported.all()) and np.all(np.asarray(near.star_leakage) > 0)
    far = read(4.5)
    assert not np.any(far.supported)
    for name in ("planet", "star_leakage", "local_zodi", "exozodi", "thermal", "n_pix"):
        np.testing.assert_array_equal(getattr(far, name), 0.0)
        np.testing.assert_array_equal(getattr(far.variance, name), 0.0)
    # The aperture is inside the support while centre + 0.7 <= 2.875, at every
    # node (the footprint at the blue node of the band is the binding one).
    blue = float(
        instrument.coronagraph.grid.frame.lod_to_arcsec(
            exposure.quadrature.nodes_nm.min()
        )
    )
    limit = (2.875 * blue / scale) - 0.7
    centres = np.linspace(0.5, 3.8, 12)
    flags = np.array([bool(read(c).supported.all()) for c in centres])
    np.testing.assert_array_equal(flags, centres <= limit + 1e-9)
    assert flags.any() and not flags.all()


@pytest.mark.medium(reason="YIP load and a 12-ring response, 6 s cold")
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
    coro = open_yip(write_yip(tmp_path / "yip")).prepare(rules=(_RULE,))
    scale = float(coro.frame.lod_to_arcsec(550.0))
    instrument = Instrument(
        primary=SimplePrimary(
            circumscribed_diameter_m=coro.frame.diameter_m, collecting_area_m2=10.0
        ),
        attenuating_elements=(ConstantThroughput(throughput=1.0),),
        coronagraph=coro,
        detector=Detector(pixel_scale_arcsec=0.1 * scale, shape=(128, 128), qe=1.0),
    )
    scene = make_system(separation_arcsec=2.6 * scale)
    exposure = cp.build_exposure(scene, instrument, (Band(549.0, 551.0),), time_d=0.0)
    response = cp.aperture_response(
        instrument,
        exposure.quadrature,
        n_separations=12,
        n_angles=8,
        max_separation_arcsec=5.5 * scale,
    )
    rings = np.asarray(response.throughput_supported)[0, 0]
    step_lod = float(response.separation_step_arcsec) / scale
    np.testing.assert_array_equal(rings, np.arange(12) * step_lod <= 3.0)
    record = cp.aperture_record(scene, instrument, exposure, response)
    inner = cp.aperture_rates(record, 2.6 * scale, 0.0, 1e-10, time_d=0.0)
    outer = cp.aperture_rates(record, 5.0 * scale, 0.0, 1e-10, time_d=0.0)
    assert bool(inner.supported.all()) and np.all(np.asarray(inner.planet) > 0)
    assert not np.any(outer.supported)
    for name in ("planet", "star_leakage", "local_zodi", "n_pix"):
        np.testing.assert_array_equal(getattr(outer, name), 0.0)


@pytest.mark.medium(reason="one 21-ring response on a 48 pixel detector, 6 s cold")
@vv.case("aperture-photon-budget", "code-verification")
def test_curves_and_record_routes_agree_on_the_flag_across_the_stamp_edge():
    """Fails if the curves route and the record route disagree on what is supported.

    Seam anchor, ``jaxedith.rates`` against ``cp.aperture_rates`` on the same
    instrument: a detector twice the table (table half-width 3 lambda/D) and a
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

    instrument = _instrument(GaussianCoronagraph(24, 0.25))
    scale = float(instrument.coronagraph.grid.frame.lod_to_arcsec(550.0))
    instrument = eqx.tree_at(
        lambda p: p.detector,
        instrument,
        Detector(pixel_scale_arcsec=0.25 * scale, shape=(48, 48), qe=1.0),
    )
    scene = make_system(separation_arcsec=1.0 * scale)
    exposure = cp.build_exposure(scene, instrument, (Band(549.0, 551.0),), time_d=0.0)
    response = cp.aperture_response(
        instrument, exposure.quadrature, n_separations=21, n_angles=8
    )
    record = cp.aperture_record(scene, instrument, exposure, response)
    curves = instrument.coronagraph.curves()
    sources = scene.sources(exposure.quadrature, exposure.resolution, 0.0)

    def flags(east_lod, north_lod):
        east, north = east_lod * scale, north_lod * scale
        image = cp.aperture_rates(record, east, north, 1e-10, time_d=0.0)
        analytic = curve_rates(
            curves,
            scene,
            instrument,
            exposure.quadrature,
            east,
            north,
            1e-10,
            time_d=0.0,
            sources=sources,
            uniform_sky=None,
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
