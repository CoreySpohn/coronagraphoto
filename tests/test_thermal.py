"""Warm-telescope emission in the image readers."""

import equinox as eqx
import jax
import numpy as np
import optixstuff as ox
from conftest import make_path, make_system
from hwoutils.constants import arcsec2rad, c, h, k_B, nm2m
from hwoutils.numerics import roundoff
from optixstuff.spectral import Band

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
    wavelength_m = np.asarray(nodes_nm, dtype=np.float64) * nm2m
    x = h * c / (wavelength_m * k_B * temperature_K)
    radiance = 2 * c / wavelength_m**4 / np.expm1(x) * nm2m
    pixel_sr = (float(instrument.detector.pixel_scale_arcsec) * arcsec2rad) ** 2
    return emissivity * radiance * float(
        instrument.primary.collecting_area_m2
    ) * pixel_sr, x


def test_thermal_image_is_the_uniform_closed_form():
    """Fails if the rendered thermal image departs from the per-pixel rate.

    Basis: Planck photon radiance, emissivity, geometric collecting area and
    pixel solid angle from host float64 primitives. The relative roundoff
    bound allows two roundings in the exponential argument, amplified by
    its magnitude, and 24 operations for Planck and radiometric products.
    The hot read allows one product and the node accumulations. Comparing
    totals also allows each component addition at the unsubtracted scale.
    The image must be uniform, carry no throughput (the path has throughput
    0.4), and add to the electron rates and the
    drawn frame. Without warm emission every field is an exact zero.
    """
    # The planet table covers the 1400 to 1500 nm band; a band that leaves a
    # table is refused when the exposure is built.
    scene = make_system(wavelengths_nm=(400.0, 700.0, 1400.0, 1500.0, 1600.0))
    cold = make_path(detector_n=16, detector_step=0.3)
    instrument = _warm(cold, 290.0, 0.5)
    exposure = cp.build_exposure(
        scene, instrument, _BANDS, time_d=0.0, plan=ox.FramePlan(100.0, 4)
    )
    rates = cp.photon_rates(scene, instrument, exposure)
    nodes = np.asarray(exposure.quadrature.nodes_nm)
    expected, exponent = _pixel_rate(instrument, nodes, 290.0, 0.5)
    assert rates.thermal.shape == (*nodes.shape, 16, 16)
    assert nodes.shape[1] > 4, "the detector field adds nodes at the default"
    np.testing.assert_allclose(
        rates.thermal,
        np.broadcast_to(expected[..., None, None], rates.thermal.shape),
        rtol=roundoff(1, n_ops=2 * exponent.max() + 24, dtype=rates.thermal.dtype),
        atol=0,
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
    n_ops = 1 + 2 * visible.quadrature.nodes_nm.size
    np.testing.assert_allclose(
        electrons.thermal,
        reference.thermal,
        rtol=roundoff(1, n_ops=n_ops, dtype=electrons.thermal.dtype),
        atol=0,
    )
    without = cp.expected(scene, cold, visible)
    scale = np.max(np.abs(electrons.total())) + np.max(np.abs(without.total()))
    np.testing.assert_allclose(
        electrons.total() - without.total(),
        electrons.thermal,
        rtol=0,
        atol=roundoff(scale, n_ops=len(electrons.component_names()) + 1),
    )
    drawn = cp.observe(jax.random.key(0), scene, hot, visible)
    assert drawn.data.shape == (1, 16, 16)
    np.testing.assert_array_equal(without.thermal, 0.0)
