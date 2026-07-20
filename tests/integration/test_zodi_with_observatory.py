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

import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import (
    ConstantThroughput,
    IdealDetector,
    OpticalPath,
    SimplePrimary,
)
from optixstuff.coronagraph import AbstractTableCoronagraph
from orbix.observatory import ObservatoryL2Halo
from skyscapes.background import LeinertZodi

from coronagraphoto.simulation import zodi_rate


class _PerfectCoronagraph(AbstractTableCoronagraph):
    """Minimal mock coronagraph: full sky transmission, no PSF dependence.

    ``zodi_rate`` only needs ``background_transmission`` (served from
    the flat ``sky_trans`` table) -- it does NOT convolve through the
    PSF datacube. So this mock is enough for the zodi pipeline; the
    stellar/PSF table members are zero stubs.
    """

    pixel_scale_lod: float
    IWA: float
    OWA: float
    psf_shape: tuple[int, int]
    sky_trans: jnp.ndarray

    def __init__(self, size: int = 65, pixel_scale_lod: float = 0.5):
        self.psf_shape = (size, size)
        self.pixel_scale_lod = pixel_scale_lod
        self.IWA = 0.0
        self.OWA = size * pixel_scale_lod / 2.0
        self.sky_trans = jnp.ones((size, size))

    def stellar_intens(self, stellar_diam_lod):
        """Zero stub (unused by the zodi pipeline)."""
        return jnp.zeros(self.psf_shape)

    def create_psfs(self, x_lod, y_lod):
        """Zero stub (unused by the zodi pipeline)."""
        k = jnp.atleast_1d(jnp.asarray(x_lod)).shape[0]
        return jnp.zeros((k, *self.psf_shape))

    def throughput(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_area(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(self, sep, wl, *, time_s=0.0):
        return 0.0

    def occulter_transmission(self, sep, wl, *, time_s=0.0):
        return 1.0

    @property
    def psf_datacube(self):
        """Unused by ``zodi_rate`` but part of the native-grid surface."""
        return None


@pytest.fixture(scope="module")
def optical_path():
    """8 m primary + perfect optics + mock coronagraph + flat detector."""
    primary = SimplePrimary(diameter_m=8.0)
    optics = ConstantThroughput(throughput=1.0)
    detector = IdealDetector(
        pixel_scale_arcsec=0.05,
        shape=(65, 65),
        quantum_efficiency=1.0,
        dark_current_rate_e_per_s=0.0,
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
        # Observability is a geometric Sun-keepout gate, not a zodi-table
        # artifact: the Leinert lookup now clamps near-Sun instead of
        # returning NaN, so mask unobservable frames by solar elongation
        # below the 45 deg Sun keepout minimum.
        elong_deg = float(obs.solar_elongation_deg(float(mjd), ra_rad, dec_rad))
        observable = elong_deg >= 45.0
        sums[i] = rate_np.sum() if observable else np.nan
    return mjds - 60575.25, sums


def test_argmax_phase_shifts_by_ecliptic_longitude(observatory, zodi, optical_path):
    """Ecliptic-plane targets at +90 deg apart peak ~90 days apart.

    The conjunction date (helio_ecliptic_longitude_deg -> 0) shifts in
    proportion to the target's ecliptic longitude. The observable maximum
    sits at the Sun-keepout edge (the closest-to-conjunction observable
    frame), so it tracks the same calendar shift.
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
    """An ecliptic-plane target's observable-peak zodi dominates a high-lat one.

    Both targets are limited to similar minimum solar elongations by the
    45 deg Sun keepout (~48 deg for the ecliptic-plane target, ~53 deg for
    the high-latitude one), so the peak is taken at the keepout edge, not
    at conjunction. There the ecliptic-plane target still dominates by
    several times because it sits in the bright zodiacal plane while the
    high-latitude target looks well above it. (Before the near-Sun clamp
    fix this ratio was >10x, but that reflected the ecliptic target
    reaching the unphysical ~15-30 deg Leinert-table edge, inside the Sun
    keepout -- not an observable epoch.)
    """
    _, sums_eq = _integrated_year(observatory, zodi, optical_path, 0.0, 0.0)
    _, sums_hi = _integrated_year(observatory, zodi, optical_path, 0.0, 60.0)
    peak_eq = float(np.nanmax(sums_eq))
    peak_hi = float(np.nanmax(sums_hi))
    assert peak_eq / peak_hi > 3.0, (
        f"Ecliptic-plane peak ({peak_eq:.2e}) should dominate the "
        f"high-latitude peak ({peak_hi:.2e}) by >3x at the keepout edge; got "
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
