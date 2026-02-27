"""Shared pytest fixtures for coronagraphoto tests."""

import jax
import pytest


@pytest.fixture
def prng_key():
    """Provide a reproducible JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def wavelength_nm():
    """Standard test wavelength (V-band)."""
    return 550.0


@pytest.fixture
def exposure_time_s():
    """Standard test exposure time (1 hour)."""
    return 3600.0


@pytest.fixture
def detector_shape():
    """Standard detector shape for tests."""
    return (100, 100)


@pytest.fixture
def pixel_scale_arcsec():
    """Standard pixel scale in arcsec/pixel."""
    return 0.01  # 10 mas/pixel
