"""Warm-telescope emission in the image readers."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
from conftest import make_path, make_system
from hwoutils.constants import arcsec2rad, c, h, k_B, nm2m
from hwoutils.numerics import roundoff
from hwoutils.testing import ulps
from optixstuff.spectral import Band

import coronagraphoto as cp

_BANDS = (Band(1400.0, 1500.0),)


def _warm(instrument, temperature_K, emissivity):
    return eqx.tree_at(
        lambda p: p.instruments[0].warm_emission,
        instrument,
        ox.WarmEmission(temperature_K, emissivity),
        is_leaf=lambda x: x is None,
    )


def _pixel_rate(instrument, nodes_nm, temperature_K, emissivity):
    """Per-pixel rate in photons/s/nm from primitives (Planck, area, pixel)."""
    wavelength_m = np.asarray(nodes_nm, dtype=np.float64) * nm2m
    x = h * c / (wavelength_m * k_B * temperature_K)
    radiance = 2 * c / wavelength_m**4 / np.expm1(x) * nm2m
    pixel_sr = (
        float(instrument.instruments[0].detector.pixel_scale_arcsec) * arcsec2rad
    ) ** 2
    return emissivity * radiance * float(
        instrument.telescope.pupil.collecting_area_m2
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
        _BANDS,
        epoch=scene.epoch,
        instrument="imager",
        time_d=0.0,
        plan=ox.FramePlan(100.0, 4),
    )
    sampling = cp.build_sampling(instrument, exposure, scene=scene)
    rates = cp.photon_rates(scene, instrument, exposure, sampling=sampling)
    nodes = np.asarray(sampling.quadrature.nodes_nm)
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
    cold_rates = cp.photon_rates(scene, cold, exposure, sampling=sampling)
    np.testing.assert_array_equal(cold_rates.thermal, 0.0)

    # The detector has quantum efficiency only in the visible, so a hot
    # emitter there shows that expected and observe include the term.
    hot = _warm(cold, 3000.0, 1.0)
    visible = cp.build_exposure(
        (Band(500.0, 600.0),),
        epoch=scene.epoch,
        instrument="imager",
        time_d=0.0,
        plan=ox.FramePlan(100.0, 4),
    )
    visible_sampling = cp.build_sampling(hot, visible, scene=scene)
    electrons = cp.expected(scene, hot, visible, sampling=visible_sampling)
    reference = hot.instruments[0].detector.electrons(
        cp.photon_rates(scene, hot, visible, sampling=visible_sampling)
    )
    assert float(electrons.thermal.min()) > 0.0
    n_ops = 1 + 2 * visible_sampling.quadrature.nodes_nm.size
    np.testing.assert_allclose(
        electrons.thermal,
        reference.thermal,
        rtol=roundoff(1, n_ops=n_ops, dtype=electrons.thermal.dtype),
        atol=0,
    )
    without = cp.expected(scene, cold, visible, sampling=visible_sampling)
    scale = np.max(np.abs(electrons.total())) + np.max(np.abs(without.total()))
    np.testing.assert_allclose(
        electrons.total() - without.total(),
        electrons.thermal,
        rtol=0,
        atol=roundoff(scale, n_ops=len(electrons.component_names()) + 1),
    )
    drawn = cp.observe(
        jax.random.key(0), scene, hot, visible, sampling=visible_sampling
    )
    assert drawn.image.data.shape == (1, 16, 16)
    np.testing.assert_array_equal(without.thermal, 0.0)


def _thermal_scene():
    # The planet table covers the 1400 to 1500 nm band.
    return make_system(wavelengths_nm=(400.0, 700.0, 1400.0, 1500.0, 1600.0))


def test_telescope_train_scales_the_sources_and_not_the_thermal_rate():
    """Fails if the shared telescope train misses a source or dims the emission.

    A half-transmitting telescope element halves the planet and the star
    leakage, which the sky photons pass through, and leaves the warm emission
    alone: its effective emissivity already includes the cold transmission.
    Truth: the factor 0.5 on the rates of the same observatory without the
    element; the four-ULP bound covers the product with the per-node gain.
    """
    scene = _thermal_scene()
    warm = _warm(make_path(detector_n=16, detector_step=0.3), 290.0, 0.5)
    dimmed = dataclasses.replace(
        warm,
        telescope=dataclasses.replace(
            warm.telescope, attenuating_elements=(ox.ConstantThroughput(0.5),)
        ),
    )
    event = cp.build_exposure(
        _BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(warm, event, scene=scene)
    full = cp.photon_rates(scene, warm, event, sampling=sampling)
    half = cp.photon_rates(scene, dimmed, event, sampling=sampling)
    for name in ("planet", "star_leakage"):
        got, reference = (
            np.asarray(getattr(half, name)),
            np.asarray(getattr(full, name)),
        )
        assert np.all(reference > 0)
        np.testing.assert_allclose(
            got, 0.5 * reference, rtol=float(ulps(4, np.ones((), got.dtype))), atol=0
        )
    np.testing.assert_array_equal(half.thermal, full.thermal)
    assert float(np.min(full.thermal)) > 0


class _HookInstrument(ox.Instrument):
    """An instrument that overrides the thermal hook and reads the telescope."""

    def thermal_photon_rate_per_sr(self, wavelength_nm, *, telescope):
        """Photons/s/nm/sr of 1e3 per square meter of the pupil."""
        area = telescope.pupil.collecting_area_m2
        return 1e3 * area * jnp.ones_like(jnp.asarray(wavelength_nm, dtype=float))


def test_a_custom_thermal_hook_is_detected_and_receives_the_telescope():
    """Fails if a hook override is ignored or is not given the telescope.

    The configuration has no warm emission, so only the override can add light.
    Truth: the override's value, 1e3 photons/s/nm/sr per square meter of the
    telescope's collecting area, times the pixel solid angle; four ULP cover
    the node-wise product.
    """
    scene = _thermal_scene()
    path = make_path(detector_n=16, detector_step=0.3)
    (configuration,) = path.instruments
    fields = {
        f.name: getattr(configuration, f.name)
        for f in dataclasses.fields(configuration)
    }
    custom = dataclasses.replace(path, instruments=(_HookInstrument(**fields),))
    event = cp.build_exposure(
        _BANDS, epoch=scene.epoch, instrument="imager", time_d=0.0
    )
    sampling = cp.build_sampling(custom, event, scene=scene)
    rates = cp.photon_rates(scene, custom, event, sampling=sampling)
    pixel_sr = (float(configuration.detector.pixel_scale_arcsec) * arcsec2rad) ** 2
    area = float(path.telescope.pupil.collecting_area_m2)
    value = 1e3 * area * pixel_sr
    np.testing.assert_allclose(
        rates.thermal,
        value,
        rtol=float(ulps(4, np.ones((), rates.thermal.dtype))),
        atol=0,
    )
    cold = cp.photon_rates(scene, path, event, sampling=sampling)
    np.testing.assert_array_equal(cold.thermal, 0.0)
