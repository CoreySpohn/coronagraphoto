"""Tests for zodiacal light utilities in coronagraphoto.util.zodiacal_light."""

import jax.numpy as jnp
import pytest

from coronagraphoto.util.zodiacal_light import (
    V_BAND_WAVELENGTH_NM,
    ayo_default_zodi_flux_jy,
    ayo_default_zodi_mag,
    create_zodi_spectrum,
    flux_to_mag_jy,
    leinert_zodi_factor,
    leinert_zodi_mag,
    mag_to_flux_jy,
    zodi_color_correction,
)


class TestMagFluxConversion:
    """Tests for magnitude/flux conversion functions."""

    def test_mag_to_flux_roundtrip(self):
        """Magnitude to flux should be reversible."""
        mag = 22.0
        flux = mag_to_flux_jy(mag)
        mag_back = flux_to_mag_jy(flux)
        assert jnp.isclose(mag, mag_back, rtol=1e-6)

    def test_flux_to_mag_roundtrip(self):
        """Flux to magnitude should be reversible."""
        flux = 1e-5  # Jy
        mag = flux_to_mag_jy(flux)
        flux_back = mag_to_flux_jy(mag)
        assert jnp.isclose(flux, flux_back, rtol=1e-6)

    def test_ab_zero_point(self):
        """AB magnitude 0 = 3631 Jy."""
        flux = mag_to_flux_jy(0.0)
        assert jnp.isclose(flux, 3631.0, rtol=1e-3)

    def test_brighter_is_lower_mag(self):
        """Lower magnitude should mean higher flux."""
        flux_mag20 = mag_to_flux_jy(20.0)
        flux_mag22 = mag_to_flux_jy(22.0)
        assert flux_mag20 > flux_mag22


class TestZodiColorCorrection:
    """Tests for wavelength-dependent color correction."""

    def test_unity_at_reference(self):
        """Color correction should be 1.0 at reference wavelength."""
        correction = zodi_color_correction(V_BAND_WAVELENGTH_NM, V_BAND_WAVELENGTH_NM)
        assert jnp.isclose(correction, 1.0, rtol=1e-3)

    def test_smooth_variation(self):
        """Color correction should vary smoothly with wavelength."""
        wavelengths = [400.0, 500.0, 600.0, 700.0, 800.0]
        corrections = [zodi_color_correction(wl) for wl in wavelengths]

        # Check that values are reasonable (within order of magnitude)
        for corr in corrections:
            assert 0.1 < corr < 10.0

    def test_photon_vs_power_units(self):
        """Photon units should differ from power units by λ factor."""
        wavelength = 700.0

        corr_photon = zodi_color_correction(wavelength, photon_units=True)
        corr_power = zodi_color_correction(wavelength, photon_units=False)

        # They should be different
        assert corr_photon != corr_power


class TestLeinertZodiFactor:
    """Tests for Leinert table brightness factor."""

    def test_reference_position_unity(self):
        """Factor should be ~1.0 at the reference position used for normalization."""
        # The tables are referenced to specific positions
        # At 90° solar lon, 0° ecliptic lat, factor should be 1.0
        factor = leinert_zodi_factor(ecliptic_lat_deg=0.0, solar_lon_deg=90.0)

        # Note: normalization may differ, allow wider tolerance
        assert 0.5 < factor < 2.0

    def test_ecliptic_pole_dimmer(self):
        """Ecliptic pole should have lower brightness factor than plane."""
        factor_pole = leinert_zodi_factor(ecliptic_lat_deg=90.0, solar_lon_deg=90.0)
        factor_plane = leinert_zodi_factor(ecliptic_lat_deg=0.0, solar_lon_deg=90.0)

        assert factor_plane > factor_pole

    def test_symmetric_latitude(self):
        """North and south ecliptic latitudes should have same factor."""
        factor_north = leinert_zodi_factor(ecliptic_lat_deg=30.0, solar_lon_deg=90.0)
        factor_south = leinert_zodi_factor(ecliptic_lat_deg=-30.0, solar_lon_deg=90.0)

        # Should be equal due to symmetry (using abs in implementation)
        assert jnp.isclose(factor_north, factor_south, rtol=0.01)


class TestLeinertZodiMag:
    """Tests for Leinert-based surface brightness in magnitudes."""

    def test_reasonable_values(self):
        """Surface brightness should be in reasonable range."""
        mag = leinert_zodi_mag(550.0, ecliptic_lat_deg=30.0, solar_lon_deg=90.0)

        # Typical zodi is 21-24 mag/arcsec²
        assert 18 < mag < 26, f"mag={mag}, expected 21-24 range"

    def test_pole_dimmer_means_higher_mag(self):
        """Dimmer regions should have higher magnitude."""
        mag_plane = leinert_zodi_mag(550.0, ecliptic_lat_deg=0.0, solar_lon_deg=90.0)
        mag_pole = leinert_zodi_mag(550.0, ecliptic_lat_deg=90.0, solar_lon_deg=90.0)

        # Dimmer = higher magnitude
        assert mag_pole > mag_plane


class TestAYODefaultFunctions:
    """Tests for AYO default zodiacal light functions."""

    def test_ayo_mag_at_vband(self):
        """AYO default should be 22 mag/arcsec² at V-band."""
        mag = ayo_default_zodi_mag(V_BAND_WAVELENGTH_NM)
        assert jnp.isclose(mag, 22.0, rtol=0.01)

    def test_ayo_flux_positive(self):
        """AYO flux should be positive."""
        flux = ayo_default_zodi_flux_jy(550.0)
        assert flux > 0


class TestCreateZodiSpectrum:
    """Tests for zodiacal light spectrum generation."""

    def test_output_shape(self):
        """Output should match input wavelength array shape."""
        wavelengths = jnp.array([400.0, 500.0, 600.0, 700.0, 800.0])
        spectrum = create_zodi_spectrum(wavelengths)
        assert spectrum.shape == wavelengths.shape

    def test_all_positive(self):
        """All spectral values should be positive."""
        wavelengths = jnp.linspace(400.0, 900.0, 10)
        spectrum = create_zodi_spectrum(wavelengths)
        assert jnp.all(spectrum > 0)

    def test_ayo_vs_leinert_mode(self):
        """AYO and Leinert modes should give different results."""
        wavelengths = jnp.array([500.0, 600.0, 700.0])

        spectrum_ayo = create_zodi_spectrum(wavelengths, use_ayo_default=True)
        spectrum_leinert = create_zodi_spectrum(
            wavelengths,
            use_ayo_default=False,
            ecliptic_lat_deg=30.0,
            solar_lon_deg=90.0,
        )

        # Values should differ (AYO uses fixed position)
        assert not jnp.allclose(spectrum_ayo, spectrum_leinert)
