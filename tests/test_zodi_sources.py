"""Tests for zodiacal light source classes in coronagraphoto.core.zodi_sources."""

import jax.numpy as jnp
import pytest

from coronagraphoto.core.zodi_sources import (
    ZodiSourceAYO,
    ZodiSourceLeinert,
    ZodiSourcePhotonFlux,
)


class TestZodiSourceAYO:
    """Tests for the AYO-compatible zodiacal light model."""

    @pytest.fixture
    def zodi_ayo(self):
        """Create a ZodiSourceAYO instance."""
        wavelengths = jnp.array([400.0, 500.0, 550.0, 600.0, 700.0, 800.0])
        return ZodiSourceAYO(wavelengths)

    def test_vband_surface_brightness(self, zodi_ayo):
        """Surface brightness at V-band should be ~22 mag/arcsec²."""
        # The default is 22 mag/arcsec² at 550 nm
        assert jnp.isclose(zodi_ayo.reference_mag_arcsec2, 22.0)

    def test_reference_wavelength(self, zodi_ayo):
        """Reference wavelength should be V-band (550 nm)."""
        assert jnp.isclose(zodi_ayo.reference_wavelength_nm, 550.0)

    def test_flux_positive(self, zodi_ayo):
        """Flux density should be positive at all wavelengths within range."""
        wavelengths = [400.0, 550.0, 700.0, 800.0]  # Stay within fixture's range
        time_jd = 2460000.0

        for wl in wavelengths:
            flux = zodi_ayo.spec_flux_density(wl, time_jd)
            assert flux > 0, f"Flux should be positive at {wl} nm"

    def test_wavelength_dependence(self, zodi_ayo):
        """Flux should vary with wavelength (solar-like spectrum)."""
        time_jd = 2460000.0

        flux_blue = zodi_ayo.spec_flux_density(400.0, time_jd)
        flux_red = zodi_ayo.spec_flux_density(700.0, time_jd)

        # Solar spectrum peaks in visible, so fluxes should differ
        assert flux_blue != flux_red

    def test_ignores_position_parameters(self, zodi_ayo):
        """AYO model uses fixed angle assumption, should ignore position."""
        time_jd = 2460000.0
        wavelength = 550.0

        flux_default = zodi_ayo.spec_flux_density(wavelength, time_jd)
        flux_with_pos = zodi_ayo.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=45.0, solar_lon_deg=90.0
        )

        # AYO model ignores these parameters
        assert jnp.isclose(flux_default, flux_with_pos)


class TestZodiSourceLeinert:
    """Tests for the Leinert table-based zodiacal light model."""

    @pytest.fixture
    def zodi_leinert(self):
        """Create a ZodiSourceLeinert instance."""
        return ZodiSourceLeinert()

    def test_reference_values(self, zodi_leinert):
        """Check default reference values."""
        assert jnp.isclose(zodi_leinert.reference_mag_arcsec2, 22.0)
        assert jnp.isclose(zodi_leinert.reference_wavelength_nm, 550.0)

    def test_flux_positive(self, zodi_leinert):
        """Flux density should be positive at valid positions."""
        time_jd = 2460000.0
        wavelength = 550.0

        flux = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=30.0, solar_lon_deg=90.0
        )
        assert flux > 0

    def test_ecliptic_pole_brighter(self, zodi_leinert):
        """Ecliptic pole should be dimmer than ecliptic plane."""
        time_jd = 2460000.0
        wavelength = 550.0
        solar_lon = 90.0

        flux_pole = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=90.0, solar_lon_deg=solar_lon
        )
        flux_plane = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=0.0, solar_lon_deg=solar_lon
        )

        # Zodi is brightest in ecliptic plane, dimmer at poles
        assert flux_plane > flux_pole

    def test_solar_longitude_dependence(self, zodi_leinert):
        """Brightness should vary with solar longitude."""
        time_jd = 2460000.0
        wavelength = 550.0
        ecliptic_lat = 30.0

        flux_90 = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=ecliptic_lat, solar_lon_deg=90.0
        )
        flux_180 = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=ecliptic_lat, solar_lon_deg=180.0
        )

        # Different positions should have different brightness
        assert flux_90 != flux_180

    def test_at_table_grid_point(self, zodi_leinert):
        """Test interpolation at exact Leinert table grid points."""
        # Table 17 has data at β=0°, λ-λ☉=90° which is a known grid point
        time_jd = 2460000.0
        wavelength = 550.0

        flux = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=0.0, solar_lon_deg=90.0
        )
        # Should return a valid value without interpolation issues
        assert jnp.isfinite(flux)
        assert flux > 0


class TestZodiSourcePhotonFlux:
    """Tests for the passthrough zodiacal light model."""

    def test_passthrough_returns_input(self):
        """Photon flux model should return the specified value."""
        wavelengths = jnp.array([500.0, 550.0, 600.0])
        photon_flux = jnp.array([1000.0, 1000.0, 1000.0])
        zodi = ZodiSourcePhotonFlux(wavelengths, photon_flux)

        time_jd = 2460000.0
        wavelength = 550.0

        flux = zodi.spec_flux_density(wavelength, time_jd)
        assert jnp.isclose(flux, 1000.0, rtol=0.01)

    def test_ignores_position_parameters(self):
        """Passthrough should ignore position parameters."""
        wavelengths = jnp.array([400.0, 600.0, 800.0])
        photon_flux = jnp.array([500.0, 500.0, 500.0])
        zodi = ZodiSourcePhotonFlux(wavelengths, photon_flux)

        # Interpolate at same wavelength with different positions
        flux1 = zodi.spec_flux_density(600.0, 2460000.0)
        flux2 = zodi.spec_flux_density(
            600.0, 2470000.0, ecliptic_lat_deg=45.0, solar_lon_deg=120.0
        )

        # Should return same value regardless of position
        assert jnp.isclose(flux1, flux2)


class TestZodiSourceComparison:
    """Cross-model comparison tests."""

    def test_ayo_leinert_similar_at_default_position(self):
        """AYO and Leinert should give similar results at ~135° solar lon."""
        wavelengths = jnp.array([500.0, 550.0, 600.0])
        zodi_ayo = ZodiSourceAYO(wavelengths)
        zodi_leinert = ZodiSourceLeinert()

        time_jd = 2460000.0
        wavelength = 550.0

        flux_ayo = zodi_ayo.spec_flux_density(wavelength, time_jd)
        # Leinert at typical coronagraph position (135° from sun, mid ecliptic)
        flux_leinert = zodi_leinert.spec_flux_density(
            wavelength, time_jd, ecliptic_lat_deg=30.0, solar_lon_deg=135.0
        )

        # Should be within a factor of ~2 (same order of magnitude)
        ratio = flux_ayo / flux_leinert
        assert 0.3 < ratio < 3.0, f"AYO/Leinert ratio={ratio:.2f}, expected ~1"
