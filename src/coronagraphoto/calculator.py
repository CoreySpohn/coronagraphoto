"""Aperture S/N and continuous exposure time through detector statistics.

Both functions read one aperture's ``optixstuff.Rates``: the record read by
:func:`coronagraphoto.aperture_rates`, or an aperture sum of
:func:`coronagraphoto.expected` images. With planet electrons/s ``p``, the
detector variance of one frame divided by its live seconds ``b``, and the
squared residual rate ``f2``, the count variance after ``t`` live seconds is
``b t + f2 t**2``, so ``S/N = p t / sqrt(b t + f2 t**2)`` and
``t = snr**2 b / (p**2 - snr**2 f2)``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array
from optixstuff import FramePlan, Rates
from optixstuff.detector import AbstractDetector


class PostProcessing(eqx.Module):
    """Residual fractions in [0, 1]; zero means perfect subtraction.

    The residual speckle and exozodi rates are these fractions of the mean
    aperture rates. Their squares add to the count variance as
    ``(fraction * rate * time)**2``; photon and electron rates are unchanged.
    """

    speckle_residual_fraction: Array = eqx.field(converter=jnp.asarray)
    exozodi_residual_fraction: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Reject concrete nonfinite or out-of-range fractions."""
        for value in (self.speckle_residual_fraction, self.exozodi_residual_fraction):
            if not isinstance(value, jax.core.Tracer) and not bool(
                jnp.all(jnp.isfinite(value) & (value >= 0) & (value <= 1))
            ):
                raise ValueError("residual fractions must be finite and in [0, 1]")


class ExposureTime(eqx.Module):
    """Continuous live seconds reaching a target S/N, one value per band.

    ``valid`` is False where no time reaches the target: the planet rate is at
    or below the residual floor, or the rates are not finite. ``time_s`` is
    zero there, so no infinity or NaN enters later arithmetic. Round to whole
    frames with ``optixstuff.frames_for_time(time_s, frame_time_s)``.
    """

    time_s: Array
    valid: Array


def _per_band(value, ndim):
    """Align a scalar or (W,) value with the band axis of ndim-dimensional rates."""
    value = jnp.asarray(value)
    if value.ndim > 1:
        raise ValueError("per-band inputs must be scalar or shape (W,)")
    return value.reshape(value.shape + (1,) * (ndim - 1)) if value.ndim else value


def _statistics(rates, detector, frame_time_s, post_processing):
    """Return planet electrons/s, variance per live second and squared floor."""
    electrons = detector.electrons(rates) if rates.carrier == "photon" else rates
    _, one_frame = detector.mean_and_variance(electrons, FramePlan(frame_time_s, 1))
    b = one_frame / _per_band(frame_time_s, one_frame.ndim)
    floor_sq = jnp.zeros_like(b)
    if post_processing is not None:
        floor_sq = (
            floor_sq
            + (
                _per_band(post_processing.speckle_residual_fraction, b.ndim)
                * electrons.star_leakage
            )
            ** 2
        )
        floor_sq = (
            floor_sq
            + (
                _per_band(post_processing.exozodi_residual_fraction, b.ndim)
                * electrons.exozodi
            )
            ** 2
        )
    return jnp.broadcast_to(electrons.planet, b.shape), b, floor_sq


def snr(
    rates: Rates,
    detector: AbstractDetector,
    time_s: ArrayLike,
    *,
    frame_time_s: ArrayLike,
    post_processing: PostProcessing | None = None,
) -> Array:
    """Return S/N per band, shape (W,), after ``time_s`` live seconds.

    ``rates`` is one aperture in photons/s/nm, to which the detector applies
    its yield here, or in electrons/s. Signal is planet electrons times live
    time. The detector supplies all independent count variance: source shot
    noise from the squared-weight ``rates.variance``, dark current, clock
    induced charge and read noise over the aperture's pixels, with CIC and
    read noise accruing once per ``frame_time_s``. ``post_processing`` adds
    the squared residual speckle and exozodi counts; None means no residual.

    ``time_s`` and ``frame_time_s`` are scalar or shape (W,). The result
    equals ``detector.mean_and_variance`` at every whole number of frames and
    interpolates between them. Zero time gives zero. NaN rates, as read
    outside an aperture record, give NaN.
    """
    p, b, floor_sq = _statistics(rates, detector, frame_time_s, post_processing)
    time_s = _per_band(time_s, p.ndim)
    variance = b * time_s + floor_sq * time_s**2
    # Only an exact zero is replaced; a NaN variance propagates.
    return p * time_s / jnp.sqrt(jnp.where(variance == 0, 1, variance))


def exposure_time(
    target_snr: ArrayLike,
    rates: Rates,
    detector: AbstractDetector,
    *,
    frame_time_s: ArrayLike,
    post_processing: PostProcessing | None = None,
) -> ExposureTime:
    """Return the continuous live time reaching each band's target S/N.

    The inverse of :func:`snr`: ``t = target_snr**2 b / (p**2 - target_snr**2
    f2)``. Read noise and CIC are exact in ``b`` because they accrue per frame
    of ``frame_time_s``. ``target_snr`` and ``frame_time_s`` are scalar or
    shape (W,).

    A positive target with ``p <= target_snr sqrt(f2)`` cannot be reached at
    any time: ``valid`` is False and ``time_s`` is zero. The same holds for
    rates that are not finite. A nonpositive target needs zero time.
    """
    p, b, floor_sq = _statistics(rates, detector, frame_time_s, post_processing)
    target_snr = _per_band(target_snr, p.ndim)
    denominator = p**2 - target_snr**2 * floor_sq
    # Feasibility compares rates, not a difference of squares, so a target
    # exactly at the residual ceiling is unreachable despite rounding.
    reachable = (
        (p > target_snr * jnp.sqrt(floor_sq)) & (denominator > 0) & jnp.isfinite(b)
    )
    time_s = (
        target_snr**2
        * jnp.where(reachable, b, 0.0)
        / jnp.where(reachable, denominator, 1.0)
    )
    trivial = (target_snr <= 0) & jnp.isfinite(p) & jnp.isfinite(b)
    return ExposureTime(jnp.where(trivial, 0.0, time_s), reachable | trivial)
