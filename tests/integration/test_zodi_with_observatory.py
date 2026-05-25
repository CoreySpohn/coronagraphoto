"""Integration: ``zodi_rate`` x ``ObservatoryL2Halo`` x ``LeinertZodi``.

Drives the full per-frame zodi-rate computation a paper figure / yield
calculation would do, with the orbix L2 halo observatory feeding
helio-ecliptic geometry into a Leinert-table-backed zodi source. Two
properties are checked over a year:

  1. Ecliptic-plane targets at different ecliptic longitudes show their
     brightness maxima at sequentially-shifted dates -- the integrated
     zodi count rate peaks when the line-of-sight passes the Sun.

  2. A high-latitude target has much weaker annual modulation than an
     ecliptic-plane target -- it never approaches conjunction.

This pins the orbix -> skyscapes -> coronagraphoto chain end-to-end.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from orbix.observatory import ObservatoryL2Halo
from skyscapes.background import LeinertZodi

from coronagraphoto import OpticalPath
from coronagraphoto.optical_elements import (
    ConstantThroughput,
    IdealDetector,
    SimplePrimary,
)
from coronagraphoto.simulation import zodi_rate


class _PerfectCoronagraph(eqx.Module):
    """Minimal mock coronagraph: full sky transmission, no PSF dependence.

    ``zodi_rate`` only needs ``sky_trans`` (flat field) and the
    detector pixel-scale conversion -- it does NOT convolve through the
    PSF datacube. So this mock is enough for the zodi pipeline.
    """

    pixel_scale_lod: float
    psf_shape: tuple[int, int]
    sky_trans: jnp.ndarray

    def __init__(self, size: int = 65, pixel_scale_lod: float = 0.5):
        self.psf_shape = (size, size)
        self.pixel_scale_lod = pixel_scale_lod
        self.sky_trans = jnp.ones((size, size))

    @property
    def psf_datacube(self):
        """Unused by ``zodi_rate`` but required by the protocol."""
        return None


@pytest.fixture(scope="module")
def optical_path():
    """8 m primary + perfect optics + mock coronagraph + flat detector."""
    primary = SimplePrimary(diameter_m=8.0)
    optics = ConstantThroughput(throughput=1.0)
    detector = IdealDetector(
        pixel_scale=0.05,
        shape=(65, 65),
        quantum_efficiency=1.0,
        dark_current_rate=0.0,
    )
    coro = _PerfectCoronagraph(size=65, pixel_scale_lod=0.5)
    return OpticalPath(primary, (optics,), coro, detector)


@pytest.fixture(scope="module")
def observatory():
    """Default L2 halo observatory at MJD 60575.25 equinox."""
    return ObservatoryL2Halo.from_default(equinox_mjd=60575.25)


@pytest.fixture(scope="module")
def zodi():
    """V-band 22 mag Leinert zodi source."""
    return LeinertZodi(reference_mag_arcsec2=22.0)


def _integrated_year(
    obs,
    zodi,
    optical_path,
    ra_deg,
    dec_deg,
    *,
    n_frames=37,
    wavelength_nm=550.0,
    bin_width_nm=50.0,
):
    """Return per-frame integrated zodi count rate (ph/s summed) over a year."""
    ra_rad = jnp.deg2rad(ra_deg)
    dec_rad = jnp.deg2rad(dec_deg)
    mjds = 60575.25 + np.linspace(0.0, 365.25, n_frames)
    sums = np.zeros(n_frames)
    for i, mjd in enumerate(mjds):
        ecl_lat = float(obs.ecliptic_latitude_deg(float(mjd), ra_rad, dec_rad))
        helio_lon = float(obs.helio_ecliptic_longitude_deg(float(mjd), ra_rad, dec_rad))
        rate = zodi_rate(
            zodi,
            optical_path,
            start_time_jd=float(mjd),
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            ecliptic_lat_deg=ecl_lat,
            solar_lon_deg=helio_lon,
        )
        rate_np = np.asarray(rate)
        # NaN at unobservable epochs -> sum=NaN; skip those.
        sums[i] = rate_np.sum() if np.isfinite(rate_np).all() else np.nan
    return mjds - 60575.25, sums


def test_argmax_phase_shifts_by_ecliptic_longitude(observatory, zodi, optical_path):
    """Ecliptic-plane targets at +90 deg apart peak ~90 days apart.

    The conjunction date (helio_ecliptic_longitude_deg -> 0) shifts in
    proportion to the target's ecliptic longitude, so the integrated
    zodi count rate's annual maximum tracks the same calendar shift.
    """
    # Target A: ecl_lon=0 (RA=0, Dec=0).
    days_a, sums_a = _integrated_year(observatory, zodi, optical_path, 0.0, 0.0)
    # Target B: ecl_lon=90 (RA=90, Dec=+23.44 compensates for obliquity).
    days_b, sums_b = _integrated_year(observatory, zodi, optical_path, 90.0, 23.44)
    argmax_a = float(days_a[np.nanargmax(sums_a)])
    argmax_b = float(days_b[np.nanargmax(sums_b)])
    shift = (argmax_b - argmax_a) % 365.25
    assert 80.0 < shift < 105.0, (
        "Conjunction should shift by ~92 days between ecl_lon=0 and "
        f"ecl_lon=90; got {shift:.1f} d"
    )


def test_ecliptic_target_brighter_than_high_latitude(observatory, zodi, optical_path):
    """An ecliptic-plane target undergoes much larger annual modulation.

    At conjunction the ecliptic-plane target's zodi is many times
    brighter than the high-latitude target's peak. Sanity check the
    ratio at each target's argmax (away from any NaN unobservable
    window).
    """
    _, sums_eq = _integrated_year(observatory, zodi, optical_path, 0.0, 0.0)
    _, sums_hi = _integrated_year(observatory, zodi, optical_path, 0.0, 60.0)
    peak_eq = float(np.nanmax(sums_eq))
    peak_hi = float(np.nanmax(sums_hi))
    assert peak_eq / peak_hi > 10.0, (
        f"Ecliptic-plane peak ({peak_eq:.2e}) should dominate the "
        f"high-latitude peak ({peak_hi:.2e}) by >10x; got "
        f"{peak_eq / peak_hi:.2f}"
    )


def test_high_latitude_target_modulation_small(observatory, zodi, optical_path):
    """High-latitude target shows <3x annual modulation -- never near Sun."""
    _, sums_hi = _integrated_year(observatory, zodi, optical_path, 0.0, 60.0)
    finite = sums_hi[np.isfinite(sums_hi)]
    assert finite.max() / finite.min() < 3.0, (
        f"High-latitude modulation should be small; got "
        f"{finite.max() / finite.min():.2f}x"
    )
