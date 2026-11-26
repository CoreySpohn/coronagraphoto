"""The response sets the order: the default field reaches every pixel.

A pixel at separation s sees the diffraction pattern slide across a band by s
times the band's fractional width in chromatic cycles, so a band needs more
nodes the farther the pixel is from the detector center. ``build_exposure``
takes its field radius from the instrument, whose default is the detector
half-diagonal, and the routes render only the nodes a band really has.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_path, make_system, with_zodi
from hwoutils.constants import Rsun2m, c, h, k_B, nm2m
from hwoutils.numerics import roundoff
from numpy.polynomial.legendre import leggauss
from optixstuff import Instrument
from optixstuff.spectral import Band
from scipy.special import j1
from skyscapes import AYOZodi
from skyscapes.scene import BlackbodyStar

import coronagraphoto as cp
from coronagraphoto import simulation

_TEFF_K = 5800.0
_CENTERS = np.linspace(450.0, 1000.0, 11)
_BANDS = tuple(Band(0.9 * c, 1.1 * c) for c in _CENTERS)
# The E8 detector: 128 by 128 pixels of 0.25 lambda/D at 550 nm.
_HALF_WIDTH_LOD = 16.0
_HALF_DIAGONAL_LOD = _HALF_WIDTH_LOD * math.sqrt(2.0)


class HalfWidthInstrument(Instrument):
    """An instrument that declares the half-width of its detector as its field."""

    def spectral_field_lod(self, band):
        """Half the longer detector side, in lambda/D at the band center."""
        ny, nx = self.detector.shape
        half_width = 0.5 * max(ny, nx) * self.detector.pixel_scale_arcsec
        return half_width / float(self.coronagraph.frame.lod_to_arcsec(band.center_nm))


def _with_half_width_field(instrument):
    fields = {
        f.name: getattr(instrument, f.name) for f in dataclasses.fields(instrument)
    }
    return HalfWidthInstrument(**fields)


def _scene():
    scene = make_system(star=BlackbodyStar(_TEFF_K, Rsun2m), smooth_planet=True)
    return with_zodi(scene, AYOZodi())


def _airy(separation_lod, center_nm, wavelength_nm):
    """Unocculted Airy response of a pixel, relative to the band center."""
    x = np.pi * separation_lod * center_nm / wavelength_nm
    return (2 * j1(x) / x) ** 2 * (center_nm / wavelength_nm) ** 2


def _fine_rule(band, cells=2000, points=8):
    unit, weight = leggauss(points)
    edges = np.linspace(band.low_nm, band.high_nm, cells + 1)
    half = np.diff(edges)[:, None] / 2
    nodes = edges[:-1, None] + half * (unit + 1)
    return nodes.ravel(), (half * weight).ravel()


def _relative_blackbody(wavelength_nm):
    """Photon Planck shape, with the common radiance factors cancelled."""
    exponent = h * c / (k_B * _TEFF_K * nm2m * wavelength_nm)
    return (550.0 / wavelength_nm) ** 4 / np.expm1(exponent)


def _worst_corner_error(exposure, instrument, *, count=67):
    """Worst relative error over the corner pixels, across every band.

    The corners are the pixels from the half-width to the half-diagonal of the
    detector, 16 to 22.6 lambda/D at 550 nm and the same angles at each band
    center. Truth is the response times the star on a 16000-point Gauss rule.
    """
    quadrature = exposure.quadrature
    worst = 0.0
    for i, band in enumerate(quadrature.bands):
        center = band.center_nm
        scale = float(instrument.coronagraph.frame.lod_to_arcsec(550.0)) / float(
            instrument.coronagraph.frame.lod_to_arcsec(center)
        )
        seps = np.linspace(_HALF_WIDTH_LOD, _HALF_DIAGONAL_LOD, count) * scale
        # The layout sets the discretization error. Reconstruct its Gauss
        # nodes on the host and verify separately what the lane actually stores.
        pieces = [
            _fine_rule(Band(lo, hi), cells=1, points=n)
            for lo, hi, n in quadrature.pieces[i]
        ]
        nodes, weights = (np.concatenate(parts) for parts in zip(*pieces, strict=True))
        for actual, reference in (
            (quadrature.nodes_nm[i], nodes),
            (quadrature.weights_nm[i], weights),
        ):
            np.testing.assert_allclose(
                actual[: nodes.size],
                reference,
                atol=0,
                rtol=roundoff(1, n_ops=4, dtype=actual.dtype),
            )
        np.testing.assert_array_equal(quadrature.weights_nm[i, nodes.size :], 0)
        star = _relative_blackbody(nodes)
        fine, fine_weights = _fine_rule(band)
        fine_star = _relative_blackbody(fine)
        for s in seps:
            got = np.sum(weights * star * _airy(s, center, nodes))
            truth = np.sum(fine_weights * fine_star * _airy(s, center, fine))
            worst = max(worst, abs(got / truth - 1))
    return worst


def test_the_corner_pixels_are_resolved_at_the_default_field():
    """Fails if the default field leaves the detector corners under-resolved.

    Truth: a blackbody star times an unocculted Airy response, integrated on a
    fine Gauss rule (2000 cells of eight points) in each of eleven 20 percent
    bands from 450 to 1000 nm, for 67 pixels from the half-width to the
    half-diagonal of the E8 detector. Tolerance 1e-8 relative at the default.
    Basis: evaluate the declared layout and independent blackbody/Airy
    integrand in host float64 so the discretization check resolves errors
    below a float32 wavelength ULP. The stored nodes and weights separately
    allow four rounding steps for the Gauss interval map and storage.
    The default field holds 1.5e-9 (measured) in that annulus;
    the bound is seven times that. An instrument that declares the half-width
    as its field misses by 1.4e-6 (nodes for 16 lambda/D, pixels at up to
    22.6), and the layout without a field (order 4) misses by 8.0e-1, so the
    corner is resolved by the field and by nothing else.
    """
    instrument = make_path(n=16, step=0.25, detector_n=128, detector_step=0.25)
    scene = _scene()
    default = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0)
    half_width = cp.build_exposure(
        scene, _with_half_width_field(instrument), _BANDS, time_d=0.0
    )
    none = cp.build_exposure(scene, instrument, _BANDS, time_d=0.0, field_lod=0.0)
    got = {
        "default": _worst_corner_error(default, instrument),
        "half_width": _worst_corner_error(half_width, instrument),
        "none": _worst_corner_error(none, instrument),
    }
    assert got["default"] < 1e-8
    assert got["half_width"] > 1e-7, "the half-width field must leave the corners short"
    assert got["none"] > 1e-1


def _unequal_setup():
    """Three bands whose zodi knots give them different orders."""
    instrument = make_path(n=16, step=0.5, detector_n=10, detector_step=0.5)
    bands = (Band(480.0, 520.0), Band(560.0, 600.0), Band(420.0, 498.0))
    return with_zodi(make_system(smooth_planet=True), AYOZodi()), instrument, bands


def _orders(quadrature):
    return [sum(n for *_, n in band) for band in quadrature.pieces]


@pytest.mark.medium(reason="renders of a three-band exposure on both routes")
def test_routes_render_the_real_nodes_only(monkeypatch):
    """Fails if a route renders padded slots, or the image changes with the padding.

    Truth: the sum of the orders of the bands' pieces, which the build records,
    is the number of node renders on each route (exact), and a band built alone
    has no padding, so the integrated image of the three-band exposure must
    equal the images of the three single-band exposures (a few ULP; tolerance
    1e-12 relative to the peak, basis floating point on the same nodes in a
    different batch layout). The padded slots of the photon record repeat the
    first real node of their band, so a ratio of two rows stays finite.
    """
    scene, instrument, bands = _unequal_setup()
    exposure = cp.build_exposure(scene, instrument, bands, time_d=0.0)
    orders = _orders(exposure.quadrature)
    width = exposure.quadrature.nodes_nm.shape[1]
    assert len(set(orders)) == 3 and max(orders) == width
    renders = []
    real_function = simulation._node_function

    def counted(*args):
        node = real_function(*args)

        def wrapped(w, i, j):
            jax.debug.callback(lambda: renders.append(1))
            return node(w, i, j)

        return wrapped

    monkeypatch.setattr(simulation, "_node_function", counted)
    for run in (cp.photon_rates, cp.expected):
        jax.clear_caches()
        renders.clear()
        result = run(scene, instrument, exposure)
        jax.effects_barrier()
        assert len(renders) == sum(orders), run.__name__
        assert result is not None
    monkeypatch.undo()

    photons = cp.photon_rates(scene, instrument, exposure)
    electrons = cp.expected(scene, instrument, exposure)
    for i, band in enumerate(bands):
        alone = cp.build_exposure(scene, instrument, (band,), time_d=0.0)
        assert alone.quadrature.nodes_nm.shape[1] == orders[i]
        photon_alone = cp.photon_rates(scene, instrument, alone).integrate()
        electron_alone = cp.expected(scene, instrument, alone)
        for name in ("star_leakage", "planet", "local_zodi"):
            a = np.asarray(getattr(photons.integrate(), name)[i])
            b = np.asarray(getattr(photon_alone, name)[0])
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * np.max(np.abs(b)))
            a = np.asarray(getattr(electrons, name)[i])
            b = np.asarray(getattr(electron_alone, name)[0])
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * np.max(np.abs(b)))
    leak = np.asarray(photons.star_leakage)
    for i, n in enumerate(orders):
        assert np.all(np.isfinite(leak[i]))
        for q in range(n, width):
            np.testing.assert_array_equal(leak[i, q], leak[i, 0])
    assert jnp.all(photons.supported)
