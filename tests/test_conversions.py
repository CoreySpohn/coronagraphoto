"""Tests for unit conversion functions in coronagraphoto.conversions."""

import jax.numpy as jnp
import pytest

from coronagraphoto import constants as const
from coronagraphoto import conversions as conv


class TestFluxConversions:
    """Tests for flux conversion functions."""

    def test_jy_to_photons_roundtrip(self):
        """Verify Jy to photons conversion is reversible."""
        wavelength_nm = jnp.array([500.0, 600.0, 700.0])
        flux_jy = jnp.array([[1.0, 1.0, 1.0]])  # 1 Jy at each wavelength

        flux_phot = conv.jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm)
        flux_jy_back = conv.photons_per_nm_per_m2_to_jy(flux_phot, wavelength_nm)

        assert jnp.allclose(flux_jy, flux_jy_back, rtol=1e-6)

    def test_jy_to_photons_wavelength_dependence(self):
        """Per-nm flux: longer wavelengths have fewer photons/nm (spread over more nm)."""
        wavelength_nm = jnp.array([400.0, 800.0])
        flux_jy = jnp.array([[1.0, 1.0]])

        flux_phot = conv.jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm)

        # Formula: flux_jy * Jy / (wavelength_nm * h)
        # As wavelength increases, photons/nm decreases
        assert flux_phot[0, 0] > flux_phot[0, 1]

    def test_mag_to_jy_conversion(self):
        """Verify AB magnitude to Jy conversion."""
        # AB magnitude 0 = 3631 Jy
        mag_0 = 0.0
        flux_jy = conv.mag_per_arcsec2_to_jy_per_arcsec2(mag_0)
        assert jnp.isclose(flux_jy, 3631.0, rtol=1e-3)

    def test_photons_to_electrons(self):
        """Verify photon to electron conversion with QE."""
        photon_rate = 1000.0
        qe = 0.9
        electron_rate = conv.photons_to_electrons(photon_rate, qe)
        assert jnp.isclose(electron_rate, 900.0)


class TestLengthConversions:
    """Tests for length conversion functions."""

    def test_nm_um_roundtrip(self):
        """Verify nm to um conversion is reversible."""
        length_nm = 550.0
        length_um = conv.nm_to_um(length_nm)
        length_nm_back = conv.um_to_nm(length_um)
        assert jnp.isclose(length_nm, length_nm_back)

    def test_nm_to_um_value(self):
        """1000 nm = 1 um."""
        assert jnp.isclose(conv.nm_to_um(1000.0), 1.0)

    def test_au_m_roundtrip(self):
        """Verify AU to m conversion is reversible."""
        length_au = 1.0
        length_m = conv.au_to_m(length_au)
        length_au_back = conv.m_to_au(length_m)
        assert jnp.isclose(length_au, length_au_back)

    def test_au_to_m_value(self):
        """1 AU should be approximately 1.496e11 m."""
        au_in_m = conv.au_to_m(1.0)
        assert jnp.isclose(au_in_m, const.AU2m, rtol=1e-6)

    def test_rearth_to_m(self):
        """1 Earth radius should be approximately 6.371e6 m."""
        r_m = conv.Rearth_to_m(1.0)
        assert jnp.isclose(r_m, const.Rearth2m, rtol=1e-3)


class TestAngularConversions:
    """Tests for angular conversion functions."""

    def test_arcsec_rad_roundtrip(self):
        """Verify arcsec to rad conversion is reversible."""
        angle_arcsec = 1.0
        angle_rad = conv.arcsec_to_rad(angle_arcsec)
        angle_arcsec_back = conv.rad_to_arcsec(angle_rad)
        assert jnp.isclose(angle_arcsec, angle_arcsec_back)

    def test_arcsec_to_rad_value(self):
        """1 arcsec = pi / (180 * 3600) rad."""
        expected = jnp.pi / (180.0 * 3600.0)
        assert jnp.isclose(conv.arcsec_to_rad(1.0), expected)

    def test_mas_arcsec_roundtrip(self):
        """Verify mas to arcsec conversion is reversible."""
        angle_mas = 100.0
        angle_arcsec = conv.mas_to_arcsec(angle_mas)
        angle_mas_back = conv.arcsec_to_mas(angle_arcsec)
        assert jnp.isclose(angle_mas, angle_mas_back)

    def test_lambda_d_arcsec_roundtrip(self):
        """Verify lambda/D to arcsec conversion is reversible."""
        wavelength_nm = 600.0
        diameter_m = 6.0
        angle_lod = 5.0

        angle_arcsec = conv.lambda_d_to_arcsec(angle_lod, wavelength_nm, diameter_m)
        angle_lod_back = conv.arcsec_to_lambda_d(
            angle_arcsec, wavelength_nm, diameter_m
        )
        assert jnp.isclose(angle_lod, angle_lod_back, rtol=1e-6)


class TestTimeConversions:
    """Tests for time conversion functions."""

    def test_years_days_roundtrip(self):
        """Verify years to days conversion is reversible."""
        time_years = jnp.array([1.0])
        time_days = conv.years_to_days(time_years)
        time_years_back = conv.days_to_years(time_days)
        assert jnp.allclose(time_years, time_years_back)

    def test_years_to_days_value(self):
        """1 year = 365.25 days."""
        assert jnp.isclose(conv.years_to_days(jnp.array([1.0]))[0], 365.25)

    def test_decimal_year_to_jd(self):
        """J2000.0 = 2451545.0 JD."""
        jd = conv.decimal_year_to_jd(jnp.array([2000.0]))
        assert jnp.isclose(jd[0], 2451545.0, rtol=1e-3)


class TestDistanceConversions:
    """Tests for distance conversion functions."""

    def test_au_arcsec_roundtrip(self):
        """Verify AU to arcsec conversion is reversible."""
        distance_au = jnp.array([1.0])
        distance_pc = 10.0

        angle_arcsec = conv.au_to_arcsec(distance_au, distance_pc)
        distance_au_back = conv.arcsec_to_au(angle_arcsec, distance_pc)
        assert jnp.allclose(distance_au, distance_au_back)

    def test_au_arcsec_at_10pc(self):
        """1 AU at 10 pc = 0.1 arcsec."""
        angle = conv.au_to_arcsec(jnp.array([1.0]), 10.0)
        assert jnp.isclose(angle[0], 0.1)


class TestMassConversions:
    """Tests for mass conversion functions."""

    def test_msun_to_kg(self):
        """1 Msun should be approximately 1.989e30 kg."""
        mass_kg = conv.Msun_to_kg(1.0)
        assert jnp.isclose(mass_kg, const.Msun2kg, rtol=1e-3)

    def test_mearth_to_kg(self):
        """1 Mearth should be approximately 5.972e24 kg."""
        mass_kg = conv.Mearth_to_kg(1.0)
        assert jnp.isclose(mass_kg, const.Mearth2kg, rtol=1e-3)
