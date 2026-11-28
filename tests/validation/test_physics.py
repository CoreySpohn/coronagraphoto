"""Detector observation statistics through the image reader."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import exposure_for, make_path, make_system, with_instrument
from optixstuff import FramePlan
from optixstuff.grids import Measure
from optixstuff.spectral import Band, Quadrature

from coronagraphoto import expected, observe


@pytest.mark.medium(reason="2000 draws plus detector sampler compilation")
def test_draw_mean_and_reproducibility():
    """Fails on noise per source, lost frame counts, or double detector yield.

    2000 seed-0 draws; five standard errors per pixel. For nine pixels,
    the normal-tail union bound is 9*5.74e-7 < 5.2e-6. Pixel means >100
    electrons make the normal approximation accurate at this draw count.
    """
    path = make_path(n=8, step=0.5, detector_n=3, detector_step=0.5)
    detector = eqx.tree_at(
        lambda d: (
            d.dark_current_e_per_pixel_s,
            d.clock_induced_charge_e_per_pixel_frame,
            d.read_noise_e_rms,
        ),
        path.instruments[0].detector,
        (jnp.array(2.0), jnp.array(0.2), jnp.array(1.0)),
    )
    path = with_instrument(path, detector=detector)
    system = make_system()
    q = Quadrature.build((Band(549.5, 550.5),), system, path.select("imager"), order=1)
    plan = FramePlan(10.0, 6.0)
    scene = system
    exposure, sampling = exposure_for(scene, q, plan=plan)
    electrons = expected(scene, path, exposure, sampling=sampling)
    mean, var = detector.mean_and_variance(electrons, plan)
    draw = eqx.filter_jit(
        lambda key: observe(key, scene, path, exposure, sampling=sampling)
    )
    key = jax.random.PRNGKey(0)
    data, again = draw(key), draw(key)
    first, second = data.image, again.image
    assert first.quantity == "electrons" and first.measure == Measure.integral()
    assert float(data.exposure.pointing.roll_deg) == 0.0
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.data, detector.draw(key, electrons, plan))
    samples = jax.lax.map(lambda k: draw(k).image.data, jax.random.split(key, 2000))
    np.testing.assert_array_less(
        np.abs(np.asarray(samples).mean(axis=0) - mean), 5 * np.sqrt(var / 2000)
    )
