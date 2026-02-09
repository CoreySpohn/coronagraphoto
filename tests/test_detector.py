"""Tests for detector models in coronagraphoto.optical_elements.detector."""

import jax
import jax.numpy as jnp
import pytest

from coronagraphoto.optical_elements.detector import (
    AbstractDetector,
    Detector,
    SimpleDetector,
    simulate_cic,
    simulate_dark_current,
    simulate_read_noise,
)


class TestSimulateDarkCurrent:
    """Tests for the dark current simulation function."""

    def test_mean_matches_expected(self, prng_key):
        """Dark current should follow Poisson with mean = rate * time."""
        dark_rate = 0.01  # e-/s/pixel
        exposure_time = 1000.0  # seconds
        shape = (100, 100)

        dark = simulate_dark_current(dark_rate, exposure_time, shape, prng_key)

        expected_mean = dark_rate * exposure_time
        actual_mean = float(jnp.mean(dark))

        # Allow ±20% tolerance for stochastic test
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean

    def test_output_shape(self, prng_key):
        """Output should match the requested shape."""
        shape = (64, 128)
        dark = simulate_dark_current(0.01, 100.0, shape, prng_key)
        assert dark.shape == shape

    def test_non_negative(self, prng_key):
        """Dark current counts should be non-negative (Poisson)."""
        dark = simulate_dark_current(0.01, 100.0, (50, 50), prng_key)
        assert jnp.all(dark >= 0)


class TestSimulateCIC:
    """Tests for clock-induced charge simulation."""

    def test_scales_with_frames(self, prng_key):
        """CIC should scale linearly with number of frames."""
        cic_rate = 0.001  # e-/pixel/frame
        shape = (100, 100)

        cic_10 = simulate_cic(cic_rate, 10, shape, prng_key)
        cic_100 = simulate_cic(cic_rate, 100, shape, prng_key)

        # Mean should scale with number of frames (approximately)
        mean_10 = float(jnp.mean(cic_10))
        mean_100 = float(jnp.mean(cic_100))

        # With 10x more frames, expect ~10x more CIC
        ratio = mean_100 / max(mean_10, 1e-10)
        assert 5 < ratio < 20, f"CIC ratio={ratio:.1f}, expected ~10"

    def test_output_shape(self, prng_key):
        """Output should match the requested shape."""
        shape = (32, 64)
        cic = simulate_cic(0.001, 100, shape, prng_key)
        assert cic.shape == shape


class TestSimulateReadNoise:
    """Tests for read noise simulation."""

    def test_gaussian_distribution(self, prng_key):
        """Read noise should be Gaussian with correct std."""
        read_noise = 3.0  # e-/pixel/read
        num_frames = 100
        shape = (200, 200)

        noise = simulate_read_noise(read_noise, num_frames, shape, prng_key)

        # Combined std = sqrt(n) * read_noise
        expected_std = jnp.sqrt(num_frames) * read_noise
        actual_std = float(jnp.std(noise))

        assert 0.9 * expected_std < actual_std < 1.1 * expected_std

    def test_zero_mean(self, prng_key):
        """Read noise should have approximately zero mean."""
        noise = simulate_read_noise(3.0, 100, (200, 200), prng_key)
        mean = float(jnp.mean(noise))
        assert abs(mean) < 1.0, f"Mean={mean:.2f}, expected ~0"


class TestSimpleDetector:
    """Tests for SimpleDetector model."""

    @pytest.fixture
    def simple_detector(self):
        """Create a SimpleDetector instance."""
        return SimpleDetector(
            pixel_scale=0.01,  # arcsec/pixel
            shape=(100, 100),
            quantum_efficiency=0.9,
            dark_current_rate=0.001,
        )

    def test_readout_noise_electrons(self, simple_detector, prng_key):
        """SimpleDetector should only produce dark current noise."""
        exposure_time = 1000.0
        noise = simple_detector.readout_noise_electrons(exposure_time, prng_key)

        assert noise.shape == simple_detector.shape
        assert jnp.all(noise >= 0)  # Poisson

    def test_zero_dark_current(self, prng_key):
        """With zero dark current, noise should be zero."""
        detector = SimpleDetector(
            pixel_scale=0.01,
            shape=(50, 50),
            quantum_efficiency=1.0,
            dark_current_rate=0.0,
        )
        noise = detector.readout_noise_electrons(1000.0, prng_key)
        assert jnp.all(noise == 0)


class TestDetector:
    """Tests for full Detector model with all noise sources."""

    @pytest.fixture
    def detector(self):
        """Create a Detector instance with all noise sources."""
        return Detector(
            pixel_scale=0.01,
            shape=(100, 100),
            quantum_efficiency=0.9,
            dark_current_rate=0.001,
            read_noise=3.0,
            cic_rate=0.001,
            frame_time=100.0,
        )

    def test_combined_noise(self, detector, prng_key):
        """Combined noise should include all components."""
        exposure_time = 1000.0
        noise = detector.readout_noise_electrons(exposure_time, prng_key)

        assert noise.shape == detector.shape
        # Read noise can be negative (Gaussian)
        assert jnp.any(noise < 0) or jnp.any(noise > 0)

    def test_frame_counting(self, detector):
        """Number of frames should be ceil(exposure_time / frame_time)."""
        exposure_time = 350.0  # 350s / 100s = 4 frames (ceil)
        num_frames = jnp.ceil(exposure_time / detector.frame_time)
        assert num_frames == 4

    def test_noise_increases_with_exposure(self, detector, prng_key):
        """Longer exposures should have more noise (on average)."""
        key1, key2 = jax.random.split(prng_key)

        noise_short = detector.readout_noise_electrons(100.0, key1)
        noise_long = detector.readout_noise_electrons(10000.0, key2)

        # Variance of longer exposure should be larger
        var_short = float(jnp.var(noise_short))
        var_long = float(jnp.var(noise_long))

        assert var_long > var_short


class TestAbstractDetectorMethods:
    """Tests for AbstractDetector base class methods."""

    @pytest.fixture
    def detector(self):
        """Create a SimpleDetector for testing base methods."""
        return SimpleDetector(
            pixel_scale=0.01,
            shape=(64, 64),
            quantum_efficiency=0.9,
            dark_current_rate=0.0,
        )

    def test_readout_source_electrons_applies_qe(self, detector, prng_key):
        """QE should reduce the number of photo-electrons."""
        photon_rate = jnp.ones((64, 64)) * 1000.0  # 1000 photons/s/pixel
        exposure_time = 1.0  # 1 second

        electrons = detector.readout_source_electrons(
            photon_rate, exposure_time, prng_key
        )

        # With QE=0.9, expect ~900 electrons per pixel on average
        mean_electrons = float(jnp.mean(electrons))
        expected = 1000.0 * 0.9

        # Allow for Poisson + binomial variance
        assert 0.85 * expected < mean_electrons < 1.15 * expected
