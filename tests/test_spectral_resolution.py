"""A rough tabulated spectrum is integrated exactly on the image path and the record.

The exposure holds the operators that reduce each table of the scene on the
band's nodes, so a stellar, planetary or zodiacal table with many narrow lines
gives the band integral of the table times the response however few nodes the
band has. The response here is a cubic in wavelength (the degree four nodes
integrate exactly), so a wrong distribution over the nodes shows; a flat
response would test only the total.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import exposure_for, make_path, make_system, with_zodi
from hwoutils.numerics import roundoff
from numpy.polynomial.legendre import leggauss
from optixstuff.grids import PixelGrid, SkyOrientation
from optixstuff.optical_elements import AbstractUniformElement
from optixstuff.spectral import Band, Quadrature
from skyscapes import PrecomputedZodi
from skyscapes.disk import AYOExozodi, ExovistaDisk
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import Planet, Star

import coronagraphoto as cp

_BAND = Band(500.0, 555.0)
_CENTER, _HALF = 527.5, 27.5


class CubicThroughput(AbstractUniformElement):
    """A positive cubic in wavelength across the band, with no declared break."""

    def get_throughput(self, wavelength_nm):
        """Return ``0.4 (1 + 0.3 u - 0.2 u^2 + 0.1 u^3)``, ``u`` across the band."""
        u = (wavelength_nm - _CENTER) / _HALF
        return 0.4 * (1 + 0.3 * u - 0.2 * u**2 + 0.1 * u**3)

    def spectral_breaks_nm(self):
        """Declare no break."""
        return np.empty(0)


def _cubic(wavelength_nm):
    return CubicThroughput().get_throughput(np.asarray(wavelength_nm))


def _lines(n, seed, depth=0.7, rate=0.25):
    """Deterministic table values with narrow absorption lines."""
    rng = np.random.default_rng(seed)
    return 1.0 - depth * (rng.random(n) < rate) * rng.random(n)


def _instrument(*, chromatic=False, detector_step=0.5):
    path = make_path(
        n=16,
        step=0.5,
        detector_n=10,
        detector_step=detector_step,
        chromatic=chromatic,
    )
    return eqx.tree_at(lambda p: p.attenuating_elements, path, (CubicThroughput(),))


def _star(tilt=0.0):
    wave = np.arange(490.0, 566.0)
    slope = 1.0 + tilt * (wave - _CENTER) / _HALF
    return Star(
        wavelengths_nm=jnp.asarray(wave),
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.asarray(3631.0 * slope * _lines(wave.size, 1))[:, None],
        interpolation_method="linear",
    )


def _flat_star(wave=None):
    if wave is None:
        wave = np.arange(490.0, 566.0)
    return Star(
        wavelengths_nm=jnp.asarray(wave),
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.full((wave.size, 1), 3631.0),
        interpolation_method="linear",
    )


def _rough_scene(zodi=None, tilt=0.0):
    """A line-rich star, an absorption-banded planet and, optionally, a zodi."""
    wave = np.arange(490.0, 566.0, 3.0)
    contrast = 1e-10 * _lines(wave.size, 2, depth=0.9)[None, :, None]
    model = GridPhysicalModel(
        jnp.asarray(wave),
        jnp.array([0.0, 90.0, 180.0]),
        jnp.asarray(np.broadcast_to(contrast, (1, wave.size, 3))),
    )
    scene = make_system(
        star=_star(tilt),
        separation_arcsec=0.03,
        planet_groups=(Planet(jnp.ones(1), model),),
    )
    return scene if zodi is None else with_zodi(scene, zodi)


def _band_integral(
    table_nm, values, response, low_nm=_BAND.low_nm, high_nm=_BAND.high_nm
):
    """Closed-form integral of a source times a response: Gauss on each cell.

    ``values`` is either the values of a linear table on ``table_nm`` or any
    function of wavelength, whose cells are then those between the knots in
    ``table_nm``. Eight Gauss points on every cell between the knots, exact
    for a linear table times a cubic, and for the smooth functions used here.
    """
    x, w = leggauss(8)
    knots = np.asarray(table_nm)
    inside = knots[(knots > low_nm) & (knots < high_nm)]
    edges = np.concatenate([[low_nm], inside, [high_nm]])
    length = np.diff(edges)[:, None]
    points = edges[:-1, None] + length * (x + 1) / 2
    source = values(points) if callable(values) else np.interp(points, knots, values)
    return float(np.sum(length * w / 2 * source * response(points)))


def _zodi(values):
    wave = np.arange(490.0, 566.0)
    return PrecomputedZodi(jnp.asarray(wave), jnp.asarray(values)), wave


def _knots(scene):
    """The merged knots of every table of the scene."""
    tables = scene.spectral_tables()
    return np.unique(np.concatenate([t.spectral_breaks_nm() for t in tables]))


def _documented_sources(scene):
    """The star and the planet as functions of wavelength, from their own reads."""

    def star(wavelength_nm):
        flux = scene.star.photon_flux_density(
            jnp.asarray(wavelength_nm), 0.0, dist_pc=scene.dist_pc
        )
        return np.asarray(flux).reshape(np.shape(wavelength_nm))

    def planet(wavelength_nm):
        flat = jnp.asarray(wavelength_nm).reshape(-1)
        flux = jax.vmap(lambda w: scene.planet_flux_densities(w, jnp.zeros(1))[0, 0])(
            flat
        )
        return np.asarray(flux).reshape(np.shape(wavelength_nm))

    return {"star_leakage": star, "planet": planet}


def _lagrange(nodes, k):
    """The Lagrange polynomial of node ``k`` as a function of wavelength."""

    def polynomial(wavelength_nm):
        out = np.ones_like(wavelength_nm)
        for j, node in enumerate(nodes):
            if j != k:
                out = out * (wavelength_nm - node) / (nodes[k] - node)
        return out

    return polynomial


def _node_effective(rule, source, knots):
    """Closed-form node-effective values of ``source`` on the first band of a rule.

    Node ``k`` of a piece gets ``(1 / w_k)`` times the integral over the piece
    of the source times the Lagrange polynomial of that node.
    """
    nodes = np.asarray(rule.nodes_nm)[0]
    weights = np.asarray(rule.weights_nm)[0]
    values, first = [], 0
    for low, high, count in rule.pieces[0]:
        piece = nodes[first : first + count]
        for k in range(count):
            integral = _band_integral(knots, source, _lagrange(piece, k), low, high)
            values.append(integral / weights[first + k])
        first += count
    return np.asarray(values)


def _images(scene, instrument, exposure):
    """Band-integrated star leakage and planet images of an exposure."""
    rates = cp.photon_rates(scene, instrument, exposure).integrate()
    return np.asarray(rates.star_leakage)[0], np.asarray(rates.planet)[0]


@pytest.mark.medium(reason="six node renders beside a render for each node")
def test_image_path_star_and_planet_are_the_node_sum_of_independent_renders():
    """Fails if the node values, their pairing with the nodes or the render are wrong.

    Code verification of every pixel of the band-integrated star leakage and
    planet images. The exposure under test is the rough scene on the default
    field and holds more than four real nodes. Its image must be the sum over
    the nodes of ``w_k * phibar_k * R_k``. The response ``R_k`` comes from
    renders that do not share that exposure: a second exposure of the same scene
    whose bands are one-node bands at the nodes of the rule under test (order 1,
    field 0), so every band has a node with index 0 and nothing in it depends on
    which image the render under test gives to which node. ``R_k`` is the node
    image of band ``k`` over that band's node value. The node-effective value
    ``phibar_k`` is computed here in closed form from the documented source
    reads: ``(1 / w_k)`` times the integral of the source times the Lagrange
    polynomial of node ``k``, eight Gauss points on every cell between the
    merged knots of the star and the planet; the node value of a one-node band
    is its band mean from the same helper. The peak-scaled roundoff bound
    covers 64 operations for node interpolation, source and response
    products, and an addition per node on each route.

    The test cannot see a response that is wrong in the same way at every node,
    since the one-node renders share it. That needs an anchor of the response
    against a closed form, which this test does not provide.
    """
    scene, instrument = _rough_scene(), _instrument(chromatic=True)
    exposure = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
    rule = exposure.quadrature
    nodes = np.asarray(rule.nodes_nm)[0]
    weights = np.asarray(rule.weights_nm)[0]
    count = sum(n for *_, n in rule.pieces[0])
    assert count > 4
    nodes, weights = nodes[:count], weights[:count]
    got = cp.photon_rates(scene, instrument, exposure).integrate()

    bands = tuple(Band(float(x) - 1e-3, float(x) + 1e-3) for x in nodes)
    one_node = Quadrature.build(bands, instrument, order=1, field_lod=0.0)
    assert one_node.nodes_nm.shape == (count, 1)
    mono = cp.photon_rates(scene, instrument, exposure_for(scene, one_node))

    knots = _knots(scene)
    for name, source in _documented_sources(scene).items():
        phibar = _node_effective(rule, source, knots)[:count]
        means = np.array(
            [
                _band_integral(
                    knots, source, lambda x: np.ones_like(x), b.low_nm, b.high_nm
                )
                / b.width_nm
                for b in bands
            ]
        )
        response = np.asarray(getattr(mono, name))[:, 0] / means[:, None, None]
        expected = np.einsum("k,kyx->yx", weights * phibar, response)
        image = np.asarray(getattr(got, name))[0]
        assert np.max(np.abs(expected)) > 0
        np.testing.assert_allclose(
            image,
            expected,
            rtol=0,
            atol=roundoff(
                np.max(np.abs(expected)), n_ops=64 + 2 * weights.size, dtype=image.dtype
            ),
        )


@pytest.mark.medium(reason="a 220-node fine rule beside the default rule")
def test_image_path_star_planet_and_contrast_match_the_fine_rule():
    """Fails if the image path samples a rough star or planet instead of integrating it.

    Truth: the same render on a rule with a piece between every knot of the star
    and the planet (about 220 nodes), for which each node sample is exact Gauss
    in the table and the reduction is the identity. The exposure under test uses
    the default field, the detector half-diagonal. The detector is 0.45
    lambda/D on 0.5 lambda/D tables: a pixel is smaller than a table cell across
    the band and no pixel center crosses a table sample, so the read samples
    the table across the band and the response is smooth in wavelength; the
    comparison then measures the rule. At 0.5 lambda/D the pixel equals a table
    cell at 550 nm, every pixel center meets a table sample there, and the
    comparison would measure that point (4.8e-4 of the peak) and not the rule.
    The guard on the fixture is asserted first.

    Per pixel, the star leakage and the planet image are held to 1e-5 of their
    peaks (measured 1.1e-8 and 8.3e-7; a build that ignores the field gives
    2.3e-5 and 2.2e-4). The detector-summed star rate, planet rate and their
    ratio, the contrast, are held to 1e-4 relative (measured 1.1e-8, 1.6e-7 and
    1.5e-7; a star sampled at the nodes misses by 3.9e-3 and a planet sampled at
    the nodes by 6e-2).
    """
    scene = _rough_scene()
    instrument = _instrument(chromatic=True, detector_step=0.45)
    table = instrument.coronagraph.grid
    cells = {}
    for wavelength in (_BAND.low_nm, _BAND.high_nm):
        pixels = instrument.detector.grid.to_lod(
            wavelength, instrument.coronagraph.frame
        )
        assert pixels.pixel_area < table.pixel_area
        cells[wavelength] = [
            np.floor(np.asarray(axis))
            for axis in table.unit_to_pixel(*pixels.coordinates())
        ]
    for low, high in zip(*cells.values(), strict=True):
        np.testing.assert_array_equal(low, high)

    coarse = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
    fine_rule = Quadrature.build((_BAND,), scene, instrument)
    assert fine_rule.nodes_nm.shape[1] > 150
    assert coarse.quadrature.nodes_nm.shape == (1, 6)
    assert "table" in {r.kind for r in coarse.resolution.rules}
    star, planet = _images(scene, instrument, coarse)
    star_truth, planet_truth = _images(
        scene, instrument, exposure_for(scene, fine_rule)
    )
    for image, truth in ((star, star_truth), (planet, planet_truth)):
        assert np.max(truth) > 0
        np.testing.assert_allclose(image, truth, rtol=0, atol=1e-5 * np.max(truth))
    np.testing.assert_allclose(star.sum(), star_truth.sum(), rtol=1e-4)
    np.testing.assert_allclose(planet.sum(), planet_truth.sum(), rtol=1e-4)
    np.testing.assert_allclose(
        planet.sum() / star.sum(), planet_truth.sum() / star_truth.sum(), rtol=1e-4
    )


def test_image_path_local_zodi_is_the_exact_table_integral():
    """Fails if the image path point-samples a rough tabulated local zodi.

    The sky transmission is uniform and the response is the cubic throughput, so
    a pixel's band-integrated zodi is a constant times the integral of the zodi
    table times the cubic. The constant is fixed by the same render with a flat
    table; truth is the closed-form integral of each table (eight Gauss points
    on every cell, exact for a linear table times a cubic). The ratio allows
    64 operations for table interpolation, radiometry and division, plus
    both node reductions.
    A point sample at the four nodes misses by order 1e-1 on this table.
    """
    instrument = _instrument()
    rough, wave = _zodi(1e12 * _lines(76, 5))
    flat, _ = _zodi(np.full(76, 1e12))
    result = {}
    for name, zodi in (("rough", rough), ("flat", flat)):
        scene = _rough_scene(zodi)
        exposure = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
        result[name] = np.asarray(
            cp.photon_rates(scene, instrument, exposure).integrate().local_zodi[0, 4, 4]
        )
    truth = _band_integral(wave, np.asarray(rough.radiance_phot_per_s_m2_nm_sr), _cubic)
    flat_truth = _band_integral(wave, np.full(76, 1e12), _cubic)
    np.testing.assert_allclose(
        result["rough"] / result["flat"],
        truth / flat_truth,
        atol=0,
        rtol=roundoff(
            1,
            n_ops=64 + 2 * exposure.quadrature.nodes_nm.size,
            dtype=result["rough"].dtype,
        ),
    )
    assert abs(truth / flat_truth - 1) > 0.05, "the table should differ from flat"


@pytest.mark.medium(reason="two scene builds and two record builds")
def test_aperture_record_zodi_read_is_the_exact_table_integral():
    """Fails if the record reads a rough zodi table at the nodes instead of reducing it.

    The record holds the response to a uniform sky and the exposure's resolution;
    a read multiplies it by the table's node-effective radiance at the read
    time. The ratio allows 64 rounding operations for node table interpolation,
    response products and division, plus both node reductions.
    """
    instrument = _instrument()
    rough, wave = _zodi(1e12 * _lines(76, 5))
    flat, _ = _zodi(np.full(76, 1e12))
    result = {}
    for name, zodi in (("rough", rough), ("flat", flat)):
        scene = _rough_scene(zodi)
        exposure = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
        response = cp.aperture_response(
            instrument,
            exposure.quadrature,
            n_separations=3,
            n_angles=3,
            max_separation_arcsec=0.04,
        )
        record = cp.aperture_record(scene, instrument, exposure, response)
        read = cp.aperture_rates(record, 0.0, 0.0, contrast=1e-10, time_d=0.0)
        assert bool(read.supported.all())
        result[name] = np.asarray(read.integrate().local_zodi[0])
    truth = _band_integral(wave, np.asarray(rough.radiance_phot_per_s_m2_nm_sr), _cubic)
    flat_truth = _band_integral(wave, np.full(76, 1e12), _cubic)
    np.testing.assert_allclose(
        result["rough"] / result["flat"],
        truth / flat_truth,
        atol=0,
        rtol=roundoff(
            1,
            n_ops=64 + 2 * exposure.quadrature.nodes_nm.size,
            dtype=result["rough"].dtype,
        ),
    )


def test_the_default_field_is_the_instruments_and_zero_keeps_the_layout():
    """Fails if the default ignores the instrument's field or ``field_lod`` is ignored.

    With no cycles a piece holds exactly ``order`` nodes, odd or even, so the
    layout at ``field_lod=0`` is the pieces times the order. The default takes
    the field the instrument declares, so it equals the build that passes that
    value as a number (identical pieces, exact) and holds more nodes than the
    order; a field of 10 lambda/D holds more than that of this small detector.
    An analytic disk contributes no breaks, so its rule and the diskless rule
    both match the direct instrument and local-zodi build bit for bit.
    """
    scene, instrument = make_system(smooth_planet=True), _instrument()
    for order in (1, 3, 4):
        exposure = cp.build_exposure(
            scene, instrument, (_BAND,), time_d=0.0, order=order, field_lod=0.0
        )
        assert exposure.quadrature.nodes_nm.shape == (1, order)
    default = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
    declared = cp.build_exposure(
        scene,
        instrument,
        (_BAND,),
        time_d=0.0,
        field_lod=instrument.spectral_field_lod(_BAND),
    )
    assert default.quadrature.pieces == declared.quadrature.pieces
    assert default.quadrature.nodes_nm.shape[1] > 4
    wide = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0, field_lod=10.0)
    assert wide.quadrature.nodes_nm.shape[1] > default.quadrature.nodes_nm.shape[1]

    zodi = PrecomputedZodi(jnp.array([490.0, 535.0, 560.0]), jnp.full(3, 1e12))
    scene = with_zodi(scene, zodi)
    with_disk = eqx.tree_at(
        lambda s: s.disk, scene, AYOExozodi(), is_leaf=lambda x: x is None
    )
    options = dict(order=4, field_lod=0.0)
    truth = Quadrature.build((_BAND,), instrument, zodi, **options)
    for system in (scene, with_disk):
        rule = cp.build_exposure(
            system, instrument, (_BAND,), time_d=0.0, **options
        ).quadrature
        np.testing.assert_array_equal(rule.nodes_nm, truth.nodes_nm)
        np.testing.assert_array_equal(rule.weights_nm, truth.weights_nm)
        assert rule.pieces == truth.pieces

    # A disk supplies breaks only, never the field: at the default field the
    # rule is still the instrument's, even for a disk that declares one.
    class WideFieldExozodi(AYOExozodi):
        def spectral_field_lod(self, band):
            return 100.0

    wide_disk = eqx.tree_at(
        lambda s: s.disk, scene, WideFieldExozodi(), is_leaf=lambda x: x is None
    )
    truth = Quadrature.build((_BAND,), instrument, zodi, order=4, field_lod=None)
    rule = cp.build_exposure(wide_disk, instrument, (_BAND,), time_d=0.0).quadrature
    np.testing.assert_array_equal(rule.nodes_nm, truth.nodes_nm)
    np.testing.assert_array_equal(rule.weights_nm, truth.weights_nm)
    assert rule.pieces == truth.pieces


def test_exovista_disk_knots_resolve_narrow_spectral_feature():
    """Integrate one disk cell against the trapezoidal knot integral.

    The sampled peak is a compact cubic B-spline whose area also equals the
    area of the piecewise-linear interpolant on these uniform knots. Four
    Gauss nodes per knot interval integrate the spline exactly. A relative
    tolerance of 1e-6 allows rounding in the contrast read and weight sum.
    """
    wave = np.arange(500.0, 601.0)
    contrast = np.full(wave.size, 1e-4)
    contrast[49:52] += 1.5 * np.array([1, 4, 1]) / 6
    grid = PixelGrid.regular((8, 8), 0.01, "arcsec", orientation=SkyOrientation(0.0))
    disk = ExovistaDisk(
        grid,
        wave,
        jnp.broadcast_to(
            jnp.asarray(contrast)[:, None, None], (wave.size, *grid.shape)
        ),
    )
    scene = make_system(star=_flat_star(wave), smooth_planet=True, disk=disk)
    exposure = cp.build_exposure(
        scene, _instrument(), (Band(500, 600),), time_d=0.0, order=4, field_lod=0.0
    )
    rule = exposure.quadrature
    nodes = rule.nodes_nm[0]
    radiance = scene.photon_radiance(nodes, exposure.time_d, grid=grid).data[:, 3, 3]
    stellar_flux = scene.star.photon_flux_density(
        nodes, exposure.time_d, dist_pc=scene.dist_pc
    )
    cell_contrast = radiance * grid.cell_area("sr") / stellar_flux.reshape(nodes.shape)
    integral = float(rule.weights_nm[0] @ cell_contrast)
    np.testing.assert_allclose(
        integral, np.trapezoid(contrast, wave), rtol=1e-6, atol=0
    )


def test_each_node_carries_its_own_star_value():
    """Fails if the node-effective star values are paired with the wrong nodes.

    Truth: the per-node ratio of the star leakage of a tilted, line-rich star to
    that of a flat star, rendered on the same four nodes, equals the ratio of the
    scene's node-effective star values (the instrument and coronagraph factors
    cancel node by node). The 32-operation relative bound covers source
    calibration, radiometric products and the two divisions. The tilt makes
    the four values differ by more than a factor of 3. Reversing the node
    rows changes the ratios by order one
    (the band-summed leakage is nearly blind to the pairing, so the test reads
    the node-resolved record).
    """
    instrument = _instrument(chromatic=True)
    tilted, flat = _rough_scene(tilt=0.9), _rough_scene()
    flat = eqx.tree_at(lambda sc: sc.star, flat, _flat_star())
    leak = {}
    for name, scene in (("tilted", tilted), ("flat", flat)):
        exposure = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
        node = np.asarray(cp.photon_rates(scene, instrument, exposure).star_leakage)
        leak[name] = node.reshape((*node.shape[:2], -1)).sum(-1)[0]
        values = scene.sources(
            exposure.quadrature, exposure.resolution, np.array([0.0])
        )
        leak[name + "_star"] = np.asarray(values.star).reshape(-1)
    assert leak["tilted_star"].max() / leak["tilted_star"].min() > 3
    np.testing.assert_allclose(
        leak["tilted"] / leak["flat"],
        leak["tilted_star"] / leak["flat_star"],
        rtol=roundoff(1, n_ops=32, dtype=leak["flat"].dtype),
        atol=0,
    )
