"""Shared pytest fixtures and data utilities for coronagraphoto tests."""

import importlib.util
import os

import jax
import pytest
from hypothesis import settings

from coronagraphoto.datasets import fetch_all, fetch_coronagraph, fetch_scene

__all__ = ["fetch_all", "fetch_coronagraph", "fetch_scene"]

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


@pytest.fixture(scope="module")
def exovista_precision():
    """Preserve subday precision for imported absolute Julian Dates."""
    with jax.enable_x64(True):
        yield


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prng_key():
    """Provide a reproducible JAX random key."""
    return jax.random.PRNGKey(0)


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
