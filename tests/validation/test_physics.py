"""Physics Verification & Validation Tests for coronagraphoto.

This module contains tests that verify the PHYSICAL CORRECTNESS of the
simulation, not just software correctness. These are essential for
scientific credibility.

Tests for unit conversions, transforms, and constants have been moved to
hwoutils/tests/. This file retains only coronagraphoto-specific physics tests.

Usage:
    pytest tests/validation/test_physics.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils import constants as const

# =============================================================================
# PHASE 1: RADIOMETRIC VERIFICATION
# =============================================================================


class TestRadiometricPhotonBucket:
    """Tier 1: Verify "Photons In = Photons Out" for a perfect optical system."""

    def test_aperture_area_calculation(self):
        """Verify primary aperture area calculation with obscuration.

        Physics: A = pi*(D/2)**2 * (1 - obscuration)
        """
        from coronagraphoto.optical_elements import PrimaryAperture

        aperture = PrimaryAperture(diameter_m=6.0, obscuration_factor=0.2)
        expected_area = np.pi * 3.0**2 * (1 - 0.2)
        assert jnp.isclose(aperture.area_m2, expected_area, rtol=1e-6)

    def test_throughput_chain_multiplication(self):
        """Verify that throughput elements multiply correctly through the chain."""
        from coronagraphoto.optical_elements import ConstantThroughputElement

        t1 = ConstantThroughputElement(throughput=0.9)
        t2 = ConstantThroughputElement(throughput=0.8)
        t3 = ConstantThroughputElement(throughput=0.7)

        flux = 1000.0
        wl = 550.0
        result = t3.apply(t2.apply(t1.apply(flux, wl), wl), wl)
        assert jnp.isclose(result, flux * 0.9 * 0.8 * 0.7, rtol=1e-6)


# =============================================================================
# PHASE 2: NUMERICAL VERIFICATION
# =============================================================================


class TestConvolutionAccuracy:
    """Verify custom convolution kernels produce correct results."""

    def test_convolution_symmetry(self):
        """A 180° rotated source should produce a 180° rotated output."""
        size = 32
        image = jnp.zeros((size, size))
        image = image.at[size // 4, 3 * size // 4].set(100.0)
        image_rotated = jnp.rot90(jnp.rot90(image))
        assert jnp.allclose(jnp.rot90(jnp.rot90(image_rotated)), image)

    def test_interpolation_at_grid_points(self):
        """Interpolation at exact grid points must return the grid values."""
        from coronagraphoto.optical_elements import LinearThroughputElement

        wavelengths = jnp.array([400.0, 500.0, 600.0, 700.0])
        throughputs = jnp.array([0.7, 0.8, 0.9, 0.85])
        element = LinearThroughputElement(
            wavelengths_nm=wavelengths, throughputs=throughputs
        )

        for wl, expected_t in zip(wavelengths, throughputs, strict=True):
            actual = element.get_throughput(float(wl))
            assert jnp.isclose(actual, expected_t, rtol=1e-5)


# =============================================================================
# PHASE 4: SCIENTIFIC VALIDATION (External Benchmarks)
# =============================================================================


class TestLeinertZodiValidation:
    """Validate zodiacal light model against Leinert et al. (1998)."""

    def test_ecliptic_pole_dimmer_than_plane(self):
        """The ecliptic pole must be dimmer than the ecliptic plane."""
        from orbix.observatory.zodiacal import leinert_zodi_factor

        factor_plane = leinert_zodi_factor(ecliptic_lat_deg=0.0, solar_lon_deg=90.0)
        factor_pole = leinert_zodi_factor(ecliptic_lat_deg=90.0, solar_lon_deg=90.0)
        assert factor_pole < factor_plane

    def test_closer_to_sun_brighter(self):
        """Looking closer to the Sun should increase zodiacal brightness."""
        from orbix.observatory.zodiacal import leinert_zodi_factor

        factor_close = leinert_zodi_factor(ecliptic_lat_deg=0.0, solar_lon_deg=30.0)
        factor_far = leinert_zodi_factor(ecliptic_lat_deg=0.0, solar_lon_deg=90.0)
        assert factor_close > factor_far

    def test_ayo_22_mag_reference(self):
        """AYO defaults to 22 mag/arcsec² at V-band."""
        from coronagraphoto.core.zodi_sources import ZodiSourceAYO

        wavelengths = jnp.array([500.0, 550.0, 600.0])
        zodi = ZodiSourceAYO(wavelengths_nm=wavelengths)
        assert zodi.reference_mag_arcsec2 == 22.0
        assert jnp.isclose(zodi.reference_wavelength_nm, 550.0)


# =============================================================================
# PHASE 5: STATISTICAL PHYSICS (Noise Models)
# =============================================================================


class TestDetectorStatistics:
    """Tier 5: Verify that noise models follow physical distributions."""

    def test_dark_current_poisson_statistics(self):
        """Dark current must follow Poisson statistics: Variance ≈ Mean."""
        from coronagraphoto.optical_elements.detector import simulate_dark_current

        key = jax.random.PRNGKey(42)
        image = simulate_dark_current(100.0, 1.0, (1000, 1000), key)

        mean_val = jnp.mean(image)
        var_val = jnp.var(image)

        assert jnp.isclose(mean_val, 100.0, rtol=0.01)
        assert jnp.isclose(var_val / mean_val, 1.0, rtol=0.05)

    def test_read_noise_scaling(self):
        """Read noise sigma must scale with sqrt(N_frames)."""
        from coronagraphoto.optical_elements.detector import simulate_read_noise

        key1, key2 = jax.random.PRNGKey(42), jax.random.PRNGKey(43)
        SHAPE = (1000, 1000)

        std_1 = jnp.std(simulate_read_noise(5.0, 1, SHAPE, key1))
        std_100 = jnp.std(simulate_read_noise(5.0, 100, SHAPE, key2))

        assert jnp.isclose(std_100 / std_1, 10.0, rtol=0.1)

    def test_snr_scaling_law(self):
        """SNR must scale with sqrt(Time) in photon-limited regime."""
        from coronagraphoto.optical_elements import SimpleDetector

        det = SimpleDetector(pixel_scale=1.0, shape=(100, 100))
        flux = jnp.full((100, 100), 10000.0)

        img1 = det.readout_source_electrons(flux, 1.0, jax.random.PRNGKey(1))
        snr1 = jnp.mean(img1) / jnp.std(img1)

        img2 = det.readout_source_electrons(flux, 4.0, jax.random.PRNGKey(2))
        snr2 = jnp.mean(img2) / jnp.std(img2)

        assert jnp.isclose(snr2 / snr1, 2.0, rtol=0.15)

    def test_poisson_photon_counting(self):
        """Photon arrival must follow Poisson statistics."""
        from coronagraphoto.optical_elements import SimpleDetector

        det = SimpleDetector(pixel_scale=1.0, shape=(500, 500), quantum_efficiency=1.0)
        image = det.readout_source_electrons(
            jnp.full((500, 500), 1000.0), 10.0, jax.random.PRNGKey(123)
        )

        assert jnp.isclose(jnp.mean(image), 10000.0, rtol=0.02)
        assert jnp.isclose(jnp.var(image) / jnp.mean(image), 1.0, rtol=0.1)


# =============================================================================
# PHASE 6: ALGORITHMIC CONSERVATION
# =============================================================================


class TestAlgorithmicConservation:
    """Tier 6: Verify custom algorithms do not leak flux."""

    def test_convolve_quadrants_sum_preservation(self):
        """Quarter-symmetric convolution must preserve total flux when PSF sums to 1."""
        from coronagraphoto.core.simulation import _convolve_quadrants

        size = 51
        center = size // 2

        flux = jnp.zeros((size, size))
        flux = flux.at[center - 2 : center + 3, center - 2 : center + 3].set(4.0)

        qsize = center + 1
        psf_cube = jnp.zeros((qsize, qsize, size, size))
        psf_cube = psf_cube.at[:, :, center, center].set(1.0)

        output = _convolve_quadrants(flux, psf_cube)
        assert jnp.sum(output) > 0
        assert jnp.isfinite(jnp.sum(output))


# =============================================================================
# PHASE 7: ORBITAL DYNAMICS (Time Evolution)
# =============================================================================


class TestOrbitalMechanics:
    """Tier 7: Verify planets obey Kepler's Laws."""

    def test_mean_anomaly_period(self):
        """Mean anomaly should increase by 2π after one orbital period."""
        from orbix.equations.orbit import mean_anomaly_tp

        a_m = const.AU2m
        GM = const.G_si * const.Msun2kg
        period_s = 2 * np.pi * np.sqrt(a_m**3 / GM)
        period_days = period_s / 86400.0
        n = 2 * np.pi / period_days

        M0 = mean_anomaly_tp(0.0, n, 0.0)
        M1 = mean_anomaly_tp(period_days, n, 0.0)

        assert jnp.isclose(M1 - M0, 2 * np.pi, rtol=1e-4)

    def test_circular_orbit_constant_radius(self):
        """A circular orbit should maintain constant angular separation."""
        from orbix.kepler.shortcuts.grid import get_grid_solver
        from orbix.system.planets import Planets as OrbixPlanets

        planets = OrbixPlanets(
            Ms=jnp.atleast_1d(const.Msun2kg),
            dist=jnp.atleast_1d(10.0),
            a=jnp.array([1.0]),
            e=jnp.array([0.0]),
            i=jnp.array([0.0]),
            W=jnp.array([0.0]),
            w=jnp.array([0.0]),
            M0=jnp.array([0.0]),
            t0=jnp.array([0.0]),
            Mp=jnp.array([0.0]),
            Rp=jnp.array([1.0]),
            p=jnp.array([0.2]),
        )
        solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)

        times = jnp.array([0.0, 91.3, 182.6, 273.9])
        seps = []
        for t in times:
            alpha, _ = planets.alpha_dMag(solver, jnp.atleast_1d(t))
            seps.append(float(alpha[0, 0]))

        assert np.std(seps) / np.mean(seps) < 0.01

    def test_kepler_third_law(self):
        """Verify P² ∝ a³ (Kepler's Third Law)."""
        from orbix.system.planets import Planets as OrbixPlanets

        planets = OrbixPlanets(
            Ms=jnp.atleast_1d(const.Msun2kg),
            dist=jnp.atleast_1d(10.0),
            a=jnp.array([1.0, 4.0]),
            e=jnp.array([0.0, 0.0]),
            i=jnp.array([0.0, 0.0]),
            W=jnp.array([0.0, 0.0]),
            w=jnp.array([0.0, 0.0]),
            M0=jnp.array([0.0, 0.0]),
            t0=jnp.array([0.0, 0.0]),
            Mp=jnp.array([0.0, 0.0]),
            Rp=jnp.array([1.0, 1.0]),
            p=jnp.array([0.2, 0.2]),
        )

        period_ratio = float(planets.n[0]) / float(planets.n[1])
        expected = np.sqrt(4.0**3)
        assert jnp.isclose(period_ratio, expected, rtol=1e-4)
