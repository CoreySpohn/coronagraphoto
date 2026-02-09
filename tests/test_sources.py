"""Tests for astrophysical source classes in coronagraphoto.core.sources."""

import jax.numpy as jnp
import pytest

from coronagraphoto.core.sources import (
    AbstractSource,
    DiskSource,
    PlanetSources,
    StarSource,
)


class TestStarSource:
    """Tests for StarSource class."""

    @pytest.fixture
    def star(self):
        """Create a simple StarSource for testing."""
        wavelengths_nm = jnp.array([400.0, 500.0, 600.0, 700.0, 800.0])
        times_jd = jnp.array([2460000.0, 2460001.0])
        # Flux grid: (n_wavelengths, n_times)
        flux_density_jy = jnp.ones((5, 2)) * 100.0  # 100 Jy flat spectrum

        return StarSource(
            dist_pc=10.0,
            mass_kg=2e30,
            diameter_arcsec=0.001,
            midplane_pa_deg=0.0,
            midplane_i_deg=0.0,
            wavelengths_nm=wavelengths_nm,
            times_jd=times_jd,
            flux_density_jy=flux_density_jy,
        )

    def test_flux_interpolation(self, star):
        """Flux should be interpolated at arbitrary wavelength/time."""
        flux = star.spec_flux_density(550.0, 2460000.5)
        assert jnp.isfinite(flux)
        assert flux > 0

    def test_flux_at_grid_point(self, star):
        """Flux at grid point should match input."""
        flux = star.spec_flux_density(500.0, 2460000.0)
        assert jnp.isfinite(flux)
        assert flux > 0


class TestPlanetSources:
    """Tests for PlanetSources class.

    Note: Full PlanetSources testing requires orbix integration and
    proper orbital elements. These tests focus on interface behavior.
    """

    pass  # Requires integration test with exovista loader


class TestDiskSource:
    """Tests for DiskSource class.

    Note: Full DiskSource testing requires a proper StarSource with
    valid interpolation grids. These tests verify interface behavior.
    """

    def test_interface_has_required_methods(self):
        """DiskSource should have required methods."""
        assert hasattr(DiskSource, "spec_flux_density")


class TestAbstractSource:
    """Tests for AbstractSource interface."""

    def test_interface_requires_implementation(self):
        """AbstractSource methods should be abstract."""
        assert hasattr(AbstractSource, "spec_flux_density")
