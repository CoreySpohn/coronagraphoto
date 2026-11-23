"""A rough tabulated spectrum is integrated exactly on the image path and the record.

The exposure holds the operators that reduce each table of the scene on the
band's nodes, so a stellar, planetary or zodiacal table with many narrow lines
gives the band integral of the table times the response however few nodes the
band has. The response here is a cubic in wavelength (the degree four nodes
integrate exactly), so a wrong distribution over the nodes shows; a flat
response would test only the total.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import exposure_for, make_path, make_system, with_zodi
from numpy.polynomial.legendre import leggauss
from optixstuff.optical_elements import AbstractUniformElement
from optixstuff.spectral import Band, Quadrature
from skyscapes import PrecomputedZodi
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


def _instrument(*, chromatic=False):
    path = make_path(
        n=16, step=0.5, detector_n=10, detector_step=0.5, chromatic=chromatic
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


def _flat_star():
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


def _band_integral(table_nm, values, response):
    """Closed-form band integral of a linear table times a cubic: Gauss on each cell."""
    x, w = leggauss(8)
    knots = np.asarray(table_nm)
    inside = knots[(knots > _BAND.low_nm) & (knots < _BAND.high_nm)]
    edges = np.concatenate([[_BAND.low_nm], inside, [_BAND.high_nm]])
    length = np.diff(edges)[:, None]
    points = edges[:-1, None] + length * (x + 1) / 2
    table = np.interp(points, knots, values)
    return float(np.sum(length * w / 2 * table * response(points)))


def _zodi(values):
    wave = np.arange(490.0, 566.0)
    return PrecomputedZodi(jnp.asarray(wave), jnp.asarray(values)), wave


@pytest.mark.medium(reason="a 200-node fine rule beside the four-node rule")
def test_image_path_star_and_planet_match_the_fine_rule():
    """Fails if the image path samples a rough star or planet instead of integrating it.

    Truth: the same render on a rule with a piece between every knot of the star
    and the planet (about 200 nodes), for which each node sample is exact Gauss
    in the table and the table reduction is the identity. The four-node rule
    with the operator must give the same band-integrated star leakage and planet
    images to 1e-3 of the peak pixel. Basis: with flat tables the same four-node
    and fine renders already differ by 3e-4 of the peak in the planet image
    (the response is a sampled pattern, not a polynomial), so the floor is the
    response and not the table operator. The floor comes from building the rule
    with the default field radius of zero (no field-dependent node refinement);
    a build that sets ``field_lod`` to the detector field gives the response
    more nodes, and the tolerance returns to 1e-4 of the peak when the default
    field radius is no longer zero.
    """
    scene, instrument = _rough_scene(), _instrument(chromatic=True)
    coarse = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
    fine_rule = Quadrature.build((_BAND,), scene, instrument)
    fine = exposure_for(scene, fine_rule)
    assert coarse.quadrature.nodes_nm.shape == (1, 4)
    assert fine_rule.nodes_nm.shape[1] > 150
    assert "table" in {r.kind for r in coarse.resolution.rules}
    got = cp.photon_rates(scene, instrument, coarse).integrate()
    truth = cp.photon_rates(scene, instrument, fine).integrate()
    for name in ("star_leakage", "planet"):
        a, b = np.asarray(getattr(got, name)), np.asarray(getattr(truth, name))
        assert np.max(np.abs(b)) > 0
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-3 * np.max(np.abs(b)))


def test_image_path_local_zodi_is_the_exact_table_integral():
    """Fails if the image path point-samples a rough tabulated local zodi.

    The sky transmission is uniform and the response is the cubic throughput, so
    a pixel's band-integrated zodi is a constant times the integral of the zodi
    table times the cubic. The constant is fixed by the same render with a flat
    table; truth is the closed-form integral of each table (eight Gauss points
    on every cell, exact for a linear table times a cubic). Tolerance 1e-10,
    basis floating point on a few dozen terms (the rule is exact to rounding).
    A point sample at the four nodes misses by order 1e-1 on this table.
    """
    instrument = _instrument()
    rough, wave = _zodi(1e12 * _lines(76, 5))
    flat, _ = _zodi(np.full(76, 1e12))
    result = {}
    for name, zodi in (("rough", rough), ("flat", flat)):
        scene = _rough_scene(zodi)
        exposure = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0)
        result[name] = float(
            cp.photon_rates(scene, instrument, exposure).integrate().local_zodi[0, 4, 4]
        )
    truth = _band_integral(wave, np.asarray(rough.radiance_phot_per_s_m2_nm_sr), _cubic)
    flat_truth = _band_integral(wave, np.full(76, 1e12), _cubic)
    np.testing.assert_allclose(
        result["rough"] / result["flat"], truth / flat_truth, rtol=1e-10
    )
    assert abs(truth / flat_truth - 1) > 0.05, "the table should differ from flat"


@pytest.mark.medium(reason="two scene builds and two record builds")
def test_aperture_record_zodi_read_is_the_exact_table_integral():
    """Fails if the record reads a rough zodi table at the nodes instead of reducing it.

    The record holds the response to a uniform sky and the exposure's resolution;
    a read multiplies it by the table's node-effective radiance at the read
    time. Same truth and constant as the image-path row, 1e-10 relative.
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
        read = cp.aperture_rates(record, 0.0, 0.0, 1e-10, time_d=0.0)
        assert bool(read.supported.all())
        result[name] = float(read.integrate().local_zodi[0])
    truth = _band_integral(wave, np.asarray(rough.radiance_phot_per_s_m2_nm_sr), _cubic)
    flat_truth = _band_integral(wave, np.full(76, 1e12), _cubic)
    np.testing.assert_allclose(
        result["rough"] / result["flat"], truth / flat_truth, rtol=1e-10
    )


def test_field_lod_reaches_the_build_and_the_default_keeps_the_layout():
    """Fails if ``field_lod`` is ignored or the default adds nodes.

    With no cycles a piece holds exactly ``order`` nodes, odd or even, so the
    default layout is the pieces times the order; a field of 10 lambda/D at the
    band centre gives the same piece more nodes. Exact counts.
    """
    scene, instrument = make_system(smooth_planet=True), _instrument()
    for order in (1, 3, 4):
        exposure = cp.build_exposure(
            scene, instrument, (_BAND,), time_d=0.0, order=order
        )
        assert exposure.quadrature.nodes_nm.shape == (1, order)
    wide = cp.build_exposure(scene, instrument, (_BAND,), time_d=0.0, field_lod=10.0)
    assert wide.quadrature.nodes_nm.shape[1] > 4


def test_each_node_carries_its_own_star_value():
    """Fails if the node-effective star values are paired with the wrong nodes.

    Truth: the per-node ratio of the star leakage of a tilted, line-rich star to
    that of a flat star, rendered on the same four nodes, equals the ratio of the
    scene's node-effective star values (the instrument and coronagraph factors
    cancel node by node). Tolerance 1e-10 relative, basis floating point on a
    product of two reads. The tilt makes the four values differ by more than a
    factor of 3, and reversing the node rows changes the ratios by order one
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
        rtol=1e-10,
    )
