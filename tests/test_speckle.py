"""Tests for speckle_rate / speckle_readout and their system wiring.

Self-contained: a scalar mock coronagraph, a mock speckle field, a duck
star, and a star-only duck scene -- no fetched datasets needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optixstuff as ox
from optixstuff.coronagraph import AbstractScalarCoronagraph

from coronagraphoto import (
    speckle_rate,
    speckle_readout,
    star_rate,
    star_readout,
    system_rate,
    system_readout,
)

_EPOCH_JD = 2451545.0  # J2000


class _MockCoro(AbstractScalarCoronagraph):
    """Scalar coronagraph with a flat stellar-leakage floor."""

    pixel_scale_lod: float = 0.5
    IWA: float = 2.0
    OWA: float = 30.0

    def throughput(self, sep, wl, *, time_s=0.0):
        return 0.5

    def core_area(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(self, sep, wl, *, time_s=0.0):
        return 1e-10

    def occulter_transmission(self, sep, wl, *, time_s=0.0):
        return 1.0

    def stellar_intens(self, stellar_diam_lod):
        return jnp.full((16, 16), 1e-9)


class _MockSpeckle(ox.AbstractSpeckleField):
    """Incoherent-halo delta with a mild, deterministic time modulation."""

    pixel_scale_lod: float = 0.5
    epoch_jd: float = _EPOCH_JD

    def realize(self, *, wavelength_nm, time_s=0.0):
        scale = 1.0 + 0.5 * jnp.cos(jnp.asarray(time_s, dtype=float))
        return jnp.full((16, 16), 1e-10) * scale


class _MockStar:
    """Duck star: a flat spectral flux density and a point-source diameter."""

    def __init__(self, flux=100.0, diameter_arcsec=0.0):
        self._flux = flux
        self.diameter_arcsec = diameter_arcsec

    def spec_flux_density(self, wavelength_nm, start_time_jd):
        return jnp.asarray(self._flux)


class _System:
    def __init__(self, star):
        self.star = star
        self.planets = ()
        self.disk = None


class _Scene:
    """Star-only duck scene (no planets / disk / zodi)."""

    def __init__(self, star):
        self.system = _System(star)
        self.zodi = None


def _optical_path(speckle=None):
    return ox.OpticalPath.from_default_setup(
        _MockCoro(),
        detector_shape=(16, 16),
        pixel_scale_arcsec=0.05,
        speckle=speckle,
    )


_OBS = dict(start_time_jd=_EPOCH_JD + 100.0, wavelength_nm=550.0, bin_width_nm=10.0)


class TestSpeckleRate:
    """The per-source speckle rate map."""

    def test_shape_and_nonnegative(self):
        """The rate map is detector-shaped, finite, and non-negative."""
        op = _optical_path(_MockSpeckle())
        m = speckle_rate(op.speckle, op, star=_MockStar(), **_OBS)
        assert m.shape == op.detector.shape
        assert jnp.all(m >= 0)
        assert jnp.all(jnp.isfinite(m))

    def test_time_varying(self):
        """The realized map changes with observation time."""
        op = _optical_path(_MockSpeckle())
        a = speckle_rate(
            op.speckle,
            op,
            star=_MockStar(),
            start_time_jd=_EPOCH_JD,
            wavelength_nm=550.0,
            bin_width_nm=10.0,
        )
        b = speckle_rate(
            op.speckle,
            op,
            star=_MockStar(),
            start_time_jd=_EPOCH_JD + 0.5,
            wavelength_nm=550.0,
            bin_width_nm=10.0,
        )
        assert float(jnp.max(jnp.abs(a - b))) > 0.0

    def test_scales_linearly_with_star_flux(self):
        """Doubling the host-star flux doubles the speckle rate."""
        op = _optical_path(_MockSpeckle())
        m1 = speckle_rate(op.speckle, op, star=_MockStar(flux=100.0), **_OBS)
        m2 = speckle_rate(op.speckle, op, star=_MockStar(flux=200.0), **_OBS)
        assert jnp.allclose(m2, 2.0 * m1, rtol=1e-5)

    def test_differentiable_in_time(self):
        """A gradient flows through the rate with respect to time."""
        op = _optical_path(_MockSpeckle())

        def total(t_jd):
            return speckle_rate(
                op.speckle,
                op,
                star=_MockStar(),
                start_time_jd=t_jd,
                wavelength_nm=550.0,
                bin_width_nm=10.0,
            ).sum()

        g = jax.grad(total)(_EPOCH_JD + 100.0)
        assert jnp.isfinite(g)


class TestSpeckleReadout:
    """The noisy speckle readout."""

    def test_shape_and_nonnegative(self):
        """The readout is detector-shaped and non-negative."""
        op = _optical_path(_MockSpeckle())
        key = jax.random.PRNGKey(0)
        m = speckle_readout(
            op.speckle, op, key, exposure_time_s=3600.0, star=_MockStar(), **_OBS
        )
        assert m.shape == op.detector.shape
        assert jnp.all(m >= 0)


class TestSystemWiring:
    """system_rate / system_readout pick up the optical-path speckle field."""

    def test_system_rate_adds_speckle(self):
        """system_rate with a speckle path equals the no-speckle sum plus it."""
        scene = _Scene(_MockStar())
        op_none = _optical_path(None)
        op_spk = _optical_path(_MockSpeckle())
        sys_args = dict(
            telescope_pa_deg=0.0, ecliptic_lat_deg=0.0, solar_lon_deg=0.0, **_OBS
        )
        base = system_rate(scene, op_none, **sys_args)
        withspk = system_rate(scene, op_spk, **sys_args)
        spk = speckle_rate(op_spk.speckle, op_spk, star=scene.system.star, **_OBS)
        assert jnp.allclose(withspk, base + spk, rtol=1e-6, atol=0.0)
        # And the speckle field genuinely contributes.
        assert float(jnp.max(jnp.abs(withspk - base))) > 0.0

    def test_system_rate_no_speckle_is_star_only(self):
        """A speckle-free star-only scene reduces to star_rate."""
        scene = _Scene(_MockStar())
        op_none = _optical_path(None)
        base = system_rate(
            scene,
            op_none,
            telescope_pa_deg=0.0,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=0.0,
            **_OBS,
        )
        star_only = star_rate(scene.system.star, op_none, **_OBS)
        assert jnp.allclose(base, star_only)

    def test_system_readout_keysplit_unchanged_without_speckle(self):
        """A speckle-free path must reproduce the pre-speckle key arithmetic."""
        scene = _Scene(_MockStar())
        op_none = _optical_path(None)
        key = jax.random.PRNGKey(3)
        sys_out = system_readout(
            scene,
            op_none,
            key,
            exposure_time_s=3600.0,
            telescope_pa_deg=0.0,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=0.0,
            **_OBS,
        )
        # Star-only, no speckle -> exactly one subkey, consumed by the star.
        k0 = jax.random.split(key, 1)[0]
        star_out = star_readout(
            scene.system.star, op_none, k0, exposure_time_s=3600.0, **_OBS
        )
        assert jnp.array_equal(sys_out, star_out)

    def test_system_readout_runs_with_speckle(self):
        """system_readout runs end-to-end with a speckle field attached."""
        scene = _Scene(_MockStar())
        op_spk = _optical_path(_MockSpeckle())
        out = system_readout(
            scene,
            op_spk,
            jax.random.PRNGKey(5),
            exposure_time_s=3600.0,
            telescope_pa_deg=0.0,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=0.0,
            **_OBS,
        )
        assert out.shape == op_spk.detector.shape
        assert jnp.all(jnp.isfinite(out))
