"""Aperture S/N and exposure-time algebra through detector statistics."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import Detector, FramePlan, Rates
from optixstuff.spectral import Band, Quadrature

import coronagraphoto as cp


def _detector():
    """Chromatic yield with every detector noise term switched on."""
    return Detector(
        shape=(8, 8),
        pixel_scale_arcsec=0.01,
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=3e-4,
        clock_induced_charge_e_per_pixel_frame=0.005,
        read_noise_e_rms=0.2,
    )


_BAND = (Band(500.0, 600.0),)


def _rates(bands=_BAND):
    """Hand-set aperture photon rates with distinct mean and variance moments."""
    q = Quadrature.build(bands, order=4)
    shape = q.nodes_nm.shape
    slope = 1 + 0.002 * (q.nodes_nm - 550.0)

    def moment(scale, n_pix):
        return dict(
            planet=scale * 2e-4 * slope,
            star_leakage=scale * jnp.full(shape, 1e-4),
            local_zodi=scale * jnp.full(shape, 3e-4),
            exozodi=scale * 2e-4 / slope,
            n_pix=jnp.full(len(bands), n_pix),
        )

    return Rates(
        q,
        "photon",
        **moment(1.0, 17.1),
        variance=Rates(q, "photon", **moment(0.8, 14.6)),
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_snr_is_detector_statistics_plus_residual_squares():
    """Fails if S/N departs from detector variance or residuals lose time squared.

    Basis: exact algebra against ``detector.mean_and_variance`` at a whole
    number of frames, with a 1e-12 relative float64 rounding allowance.
    """
    rates, detector = _rates(), _detector()
    e = detector.electrons(rates)
    plan = FramePlan(100.0, 36)
    _, base = detector.mean_and_variance(e, plan)
    for f, g in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.2)):
        actual = cp.snr(
            rates,
            detector,
            3600.0,
            frame_time_s=100.0,
            post_processing=cp.PostProcessing(f, g),
        )
        expected = base + (f * e.star_leakage * 3600) ** 2 + (g * e.exozodi * 3600) ** 2
        np.testing.assert_allclose(
            (e.planet * 3600 / actual) ** 2, expected, rtol=1e-12, atol=0
        )
    kwargs = dict(frame_time_s=100.0)
    np.testing.assert_allclose(
        cp.snr(rates, detector, 3600.0, **kwargs),
        e.planet * 3600 / jnp.sqrt(base),
        rtol=1e-12,
    )
    # Electron rates are read as they are; zero time gives zero, not NaN.
    np.testing.assert_allclose(
        cp.snr(e, detector, 3600.0, **kwargs),
        cp.snr(rates, detector, 3600.0, **kwargs),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(cp.snr(rates, detector, 0.0, **kwargs), [0.0])
    for bad in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cp.PostProcessing(bad, 0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cp.PostProcessing(0, bad)


@vv.case("aperture-photon-budget", "code-verification")
def test_exposure_time_inverts_snr_and_flags_unreachable_targets():
    """Fails if the solved time misses the target or an unreachable one looks valid.

    Basis: exact inverse algebra, 1e-12 relative float64 rounding allowance;
    the derivative is compared with a central difference of step 1e-4, whose
    O(h^2) truncation allows 1e-7 relative.
    """
    rates, detector = _rates(), _detector()
    pp = cp.PostProcessing(0.1, 0.2)
    kwargs = dict(frame_time_s=100.0, post_processing=pp)
    times = jnp.array([100.0, 723.4, 3600.0, 1e5])
    targets = jax.vmap(lambda t: cp.snr(rates, detector, t, **kwargs)[0])(times)

    def solve(target):
        return cp.exposure_time(target, rates, detector, **kwargs)

    solved = jax.jit(jax.vmap(solve))(targets)
    np.testing.assert_allclose(solved.time_s[:, 0], times, rtol=1e-12)
    assert solved.valid.all()

    e = detector.electrons(rates)
    ceiling = e.planet / (0.1 * e.star_leakage)
    assert cp.snr(rates, detector, 1e30, **kwargs)[0] < ceiling[0]
    for target in (ceiling * 1.001, ceiling, jnp.sqrt(ceiling**2 * 1.0)):
        beyond = solve(target)
        assert not beyond.valid.any()
        np.testing.assert_array_equal(beyond.time_s, [0.0])
    nothing = solve(0.0)
    assert nothing.valid.all()
    np.testing.assert_array_equal(nothing.time_s, [0.0])

    step = 1e-4
    slope = jax.grad(lambda t: solve(t).time_s[0])(5.0)
    difference = (solve(5.0 + step).time_s[0] - solve(5.0 - step).time_s[0]) / (
        2 * step
    )
    np.testing.assert_allclose(slope, difference, rtol=1e-7)
    # An unreachable target has a finite (zero) derivative, not NaN.
    assert jax.grad(lambda t: solve(t).time_s[0])(ceiling[0] * 2) == 0.0


@vv.case("aperture-photon-budget", "code-verification")
def test_unknown_rates_stay_unknown():
    """Fails if a NaN rate reaches S/N or a time as a valid-looking number.

    Basis: exact; NaN must propagate to S/N and clear ``valid``.
    """
    rates, detector = _rates(), _detector()
    nan = jnp.full(rates.quadrature.nodes_nm.shape, jnp.nan)
    # The signal comes from the mean record and the noise from its variance.
    for where in (lambda r: r.planet, lambda r: r.variance.local_zodi):
        unknown = eqx.tree_at(where, rates, nan)
        assert jnp.isnan(cp.snr(unknown, detector, 3600.0, frame_time_s=100.0)).all()
        solved = cp.exposure_time(5.0, unknown, detector, frame_time_s=100.0)
        assert not solved.valid.any()
        np.testing.assert_array_equal(solved.time_s, [0.0])


@vv.case("aperture-photon-budget", "code-verification")
def test_per_band_targets_and_frame_times():
    """Fails if (W,) targets, times or frame times mix bands.

    Basis: each band of a two-band call equals its own one-band call to a
    1e-12 relative rounding allowance.
    """
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    rates, detector = _rates(bands), _detector()
    pp = cp.PostProcessing(0.05, 0.1)
    target, frame, time = jnp.array([4.0, 6.0]), jnp.array([50.0, 200.0]), 5000.0
    both = cp.exposure_time(
        target, rates, detector, frame_time_s=frame, post_processing=pp
    )
    snrs = cp.snr(rates, detector, time, frame_time_s=frame, post_processing=pp)
    assert both.time_s.shape == snrs.shape == (2,)
    for i, band in enumerate(bands):
        one = _rates((band,))
        single = cp.exposure_time(
            target[i], one, detector, frame_time_s=frame[i], post_processing=pp
        )
        np.testing.assert_allclose(both.time_s[i], single.time_s[0], rtol=1e-12)
        np.testing.assert_allclose(
            snrs[i],
            cp.snr(one, detector, time, frame_time_s=frame[i], post_processing=pp)[0],
            rtol=1e-12,
        )
    with pytest.raises(ValueError, match="scalar or shape"):
        cp.snr(rates, detector, jnp.ones((2, 2)), frame_time_s=100.0)
