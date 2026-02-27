"""Integration Tests with Real Coronagraph Data.

These tests verify that coronagraphoto can load and interact with real
coronagraph data from yippy. They focus on testing the data loading and
basic attribute access rather than full simulation chain (which has
known JAX/numpy compatibility issues).

Tests are marked slow and require network access.
"""

import numpy as np
import pytest
from lod_unit import lod
from yippy import Coronagraph

from coronagraphoto.datasets import fetch_coronagraph

# Mark all tests as slow (network access) and integration
pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def coro_path():
    """Fetch coronagraph path once per module."""
    return fetch_coronagraph()


@pytest.fixture(scope="module")
def real_coronagraph(coro_path):
    """Load the real eac1_aavc_512 coronagraph from cached data."""
    return Coronagraph(coro_path)


class TestCoronagraphLoading:
    """Tests for loading and inspecting real coronagraph data."""

    def test_coronagraph_path_exists(self, coro_path):
        """Verify the coronagraph path points to a valid directory."""
        import os

        assert os.path.isdir(coro_path), f"Coronagraph path not found: {coro_path}"

    def test_coronagraph_loads_successfully(self, real_coronagraph):
        """Verify coronagraph data loads without errors."""
        assert real_coronagraph is not None

    def test_coronagraph_has_stellar_intens(self, real_coronagraph):
        """Verify coronagraph has stellar intensity interface."""
        assert hasattr(real_coronagraph, "stellar_intens")

    def test_coronagraph_has_create_psfs(self, real_coronagraph):
        """Verify coronagraph has off-axis PSF creation interface."""
        assert hasattr(real_coronagraph, "offax")
        assert hasattr(real_coronagraph.offax, "create_psfs")

    def test_coronagraph_has_pixel_scale(self, real_coronagraph):
        """Verify coronagraph has pixel scale attribute."""
        assert hasattr(real_coronagraph, "pixel_scale")
        # Should be around 0.25 lod/pixel for aavc
        assert 0.1 < real_coronagraph.pixel_scale.value < 1.0


class TestStellarIntensity:
    """Tests for stellar intensity (on-axis) PSF retrieval."""

    def test_stellar_intens_point_source(self, real_coronagraph):
        """Verify stellar intensity works for point source (0 diameter)."""
        psf = real_coronagraph.stellar_intens(0.0 * lod)
        assert psf is not None
        assert psf.ndim == 2, f"Expected 2D PSF, got {psf.ndim}D"
        assert psf.shape[0] > 0 and psf.shape[1] > 0

    def test_stellar_intens_is_normalized(self, real_coronagraph):
        """Verify stellar PSF is approximately normalized."""
        psf = real_coronagraph.stellar_intens(0.0 * lod)
        total = np.sum(psf)
        # Coronagraph PSFs suppress starlight, so sum should be << 1
        # but still positive
        assert total > 0, "PSF sum is zero or negative"
        assert total < 1.0, f"PSF sum {total} > 1 (not suppressed?)"

    def test_stellar_intens_small_diameter(self, real_coronagraph):
        """Verify stellar intensity works for small (non-zero) diameter."""
        psf = real_coronagraph.stellar_intens(0.01 * lod)
        assert psf is not None
        assert np.all(np.isfinite(psf)), "PSF contains non-finite values"

    def test_stellar_intens_larger_diameter(self, real_coronagraph):
        """Verify stellar intensity works for larger resolved source."""
        psf = real_coronagraph.stellar_intens(0.1 * lod)
        assert psf is not None
        # Larger star should leak more light through
        psf_point = real_coronagraph.stellar_intens(0.0 * lod)
        assert np.sum(psf) >= np.sum(psf_point) * 0.9  # Allow some tolerance


class TestOffAxisPSFs:
    """Tests for off-axis (planet) PSF creation."""

    def test_create_psfs_at_single_position(self, real_coronagraph):
        """Verify PSF creation works for single off-axis position."""
        x_lod = np.array([5.0]) * lod
        y_lod = np.array([0.0]) * lod
        psfs = real_coronagraph.offax.create_psfs(x_lod, y_lod)
        assert psfs is not None
        assert psfs.ndim == 3, f"Expected batch of 2D PSFs, got {psfs.ndim}D"
        assert psfs.shape[0] == 1, "Expected 1 PSF"

    def test_create_psfs_at_multiple_positions(self, real_coronagraph):
        """Verify PSF creation works for multiple positions."""
        x_lod = np.array([3.0, 5.0, 10.0]) * lod
        y_lod = np.array([0.0, 0.0, 0.0]) * lod
        psfs = real_coronagraph.offax.create_psfs(x_lod, y_lod)
        assert psfs.shape[0] == 3, f"Expected 3 PSFs, got {psfs.shape[0]}"

    def test_offaxis_psfs_are_normalized(self, real_coronagraph):
        """Verify off-axis PSFs are approximately normalized."""
        x_lod = np.array([5.0]) * lod
        y_lod = np.array([0.0]) * lod
        psfs = real_coronagraph.offax.create_psfs(x_lod, y_lod)
        total = np.sum(psfs[0])
        # Off-axis PSFs should preserve most flux (high throughput region)
        assert total > 0.1, f"PSF sum {total} too low (expected > 0.1)"
        assert total < 1.5, f"PSF sum {total} too high (expected < 1.5)"


class TestCoronagraphProperties:
    """Tests for coronagraph properties and curves."""

    def test_has_iwa(self, real_coronagraph):
        """Verify coronagraph has inner working angle."""
        assert hasattr(real_coronagraph, "IWA")
        # Should be a few lambda/D
        assert real_coronagraph.IWA.value > 0

    def test_has_owa(self, real_coronagraph):
        """Verify coronagraph has outer working angle."""
        assert hasattr(real_coronagraph, "OWA")
        # OWA should be larger than IWA
        assert real_coronagraph.OWA > real_coronagraph.IWA

    def test_has_throughput_curve(self, real_coronagraph):
        """Verify coronagraph has throughput curve."""
        assert hasattr(real_coronagraph, "throughput")
        # Should be callable or have values
        assert real_coronagraph.throughput is not None
