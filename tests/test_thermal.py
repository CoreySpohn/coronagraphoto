"""Warm-telescope emission in the image readers."""

import equinox as eqx
import jax
import numpy as np
import optixstuff as ox
from conftest import make_path, make_system
from optixstuff.spectral import Band
from scipy import constants

import coronagraphoto as cp

_BANDS = (Band(1400.0, 1500.0),)


def _warm(instrument, temperature_K, emissivity):
    return eqx.tree_at(
        lambda p: p.warm_emission,
        instrument,
        ox.WarmEmission(temperature_K, emissivity),
        is_leaf=lambda x: x is None,
    )


def _pixel_rate(instrument, nodes_nm, temperature_K, emissivity):
    """Per-pixel rate in photons/s/nm from primitives (Planck, area, pixel)."""
    wavelength_m = np.asarray(nodes_nm) * 1e-9
    x = constants.h * constants.c / (wavelength_m * constants.k * temperature_K)
    radiance = 2 * constants.c / wavelength_m**4 / np.expm1(x) * 1e-9
    pixel_sr = np.deg2rad(instrument.detector.pixel_scale_arcsec / 3600.0) ** 2
    return emissivity * radiance * instrument.primary.collecting_area_m2 * pixel_sr


def test_thermal_image_is_the_uniform_closed_form():
    """Fails if the rendered thermal image departs from the per-pixel rate.

    Basis: Planck photon radiance, emissivity, geometric collecting area and
    pixel solid angle from primitives, float64, rtol 1e-12 (an exponential and
    a four-factor product). The image must be uniform, carry no throughput
    (the path has throughput 0.4), and add to the electron rates and the
    drawn frame. Without warm emission every field is an exact zero.
    """
    scene = make_system()
    cold = make_path(detector_n=16, detector_step=0.3)
    instrument = _warm(cold, 290.0, 0.5)
    exposure = cp.build_exposure(
        scene, instrument, _BANDS, time_d=0.0, plan=ox.FramePlan(100.0, 4)
    )
    rates = cp.photon_rates(scene, instrument, exposure)
    nodes = np.asarray(exposure.quadrature.nodes_nm)
    expected = _pixel_rate(instrument, nodes, 290.0, 0.5)
    assert rates.thermal.shape == (1, 4, 16, 16)
    np.testing.assert_allclose(
        rates.thermal,
        np.broadcast_to(expected[..., None, None], rates.thermal.shape),
        rtol=1e-12,
    )
    assert np.all(expected > 0)
    cold_rates = cp.photon_rates(scene, cold, exposure)
    np.testing.assert_array_equal(cold_rates.thermal, 0.0)

    # The detector has quantum efficiency only in the visible, so a hot
    # emitter there shows that expected and observe include the term.
    hot = _warm(cold, 3000.0, 1.0)
    visible = cp.build_exposure(
        scene, hot, (Band(500.0, 600.0),), time_d=0.0, plan=ox.FramePlan(100.0, 4)
    )
    electrons = cp.expected(scene, hot, visible)
    reference = hot.detector.electrons(cp.photon_rates(scene, hot, visible))
    assert float(electrons.thermal.min()) > 0.0
    np.testing.assert_allclose(electrons.thermal, reference.thermal, rtol=1e-12)
    without = cp.expected(scene, cold, visible)
    np.testing.assert_allclose(
        electrons.total() - without.total(), electrons.thermal, rtol=1e-9
    )
    drawn = cp.observe(jax.random.key(0), scene, hot, visible)
    assert drawn.data.shape == (1, 16, 16)
    np.testing.assert_array_equal(without.thermal, 0.0)
