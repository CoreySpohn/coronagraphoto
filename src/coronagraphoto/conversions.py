"""Unit conversion functions using centralized constants.

Note: Functions are NOT JIT-compiled to allow JAX to fuse them into larger kernels.
JIT-compile the top-level functions that use these conversions.
"""

import jax.numpy as jnp

from coronagraphoto import constants as const


# Flux conversions
def jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm):
    """Convert flux density from Janskys to photons/s/nm/m^2.

    Args:
        flux_jy:
            The flux density in Janskys. Assumed to be a 2D array of
            (wavelength, time).
        wavelength_nm:
            The wavelength in nanometers. Assumed to be a 1D array.

    Returns:
            The flux density in photons/s/nm/m^2.
    """
    # Units here cancel out to give photons/s/nm/m^2
    return flux_jy * const.Jy / (wavelength_nm * const.h)


def photons_per_nm_per_m2_to_jy(flux_phot, wavelength_nm):
    """Convert flux density from photons/s/nm/m^2 to Janskys.

    Args:
        flux_phot:
            The flux density in photons/s/nm/m^2. Assumed to be a 2D
            array of (wavelength, time).
        wavelength_nm:
            The wavelength in nanometers. Assumed to be a 1D array.

    Returns:
            The flux density in Janskys.
    """
    return flux_phot * (wavelength_nm * const.h) / const.Jy


def mag_per_arcsec2_to_jy_per_arcsec2(mag_per_arcsec2):
    """Convert surface brightness from mag/arcsec^2 to Jy/arcsec^2 (AB).

    Args:
        mag_per_arcsec2: Surface brightness in magnitudes per arcsec squared.

    Returns:
        Surface brightness in Janskys per arcsec squared.
    """
    # AB magnitude zero point is 3631 Jy
    f0_jy = 3631.0
    return f0_jy * 10 ** (-0.4 * mag_per_arcsec2)


def photons_to_electrons(photon_rate, quantum_efficiency):
    """Convert photon rate to electron rate.

    Args:
        photon_rate: Photons per second
        quantum_efficiency: QE of detector (0-1)

    Returns:
        Electrons per second
    """
    return photon_rate * quantum_efficiency


# Length conversions
def nm_to_um(length_nm):
    """Convert length from nanometers to micrometers."""
    return length_nm * const.nm2um


def um_to_nm(length_um):
    """Convert length from micrometers to nanometers."""
    return length_um * const.um2nm


def au_to_m(length_au):
    """Convert length from AU to meters."""
    return length_au * const.AU2m


def m_to_au(length_m):
    """Convert length from meters to AU."""
    return length_m * const.m2AU


def Rearth_to_m(length_Rearth):
    """Convert length from Earth radii to meters."""
    return length_Rearth * const.Rearth2m


# Velocity conversions
def au_per_yr_to_m_per_s(velocity_au_per_yr):
    """Convert velocity from AU/yr to m/s."""
    return velocity_au_per_yr * const.AU2m / const.yr2s


# Angular conversions
def arcsec_to_rad(angle_arcsec):
    """Convert angle from arcseconds to radians."""
    return angle_arcsec * const.arcsec2rad


def rad_to_arcsec(angle_rad):
    """Convert angle from radians to arcseconds."""
    return angle_rad * const.rad2arcsec


def mas_to_arcsec(angle_mas):
    """Convert angle from milliarcseconds to arcseconds."""
    return angle_mas * const.mas2arcsec


def arcsec_to_mas(angle_arcsec):
    """Convert angle from arcseconds to milliarcseconds."""
    return angle_arcsec * const.arcsec2mas


def arcsec_to_lambda_d(angle_arcsec, wavelength_nm, diameter_m):
    """Convert angular separation to lambda/D units.

    Args:
        angle_arcsec: Angular separation in arcseconds
        wavelength_nm: Wavelength in nanometers
        diameter_m: Telescope diameter in meters

    Returns:
        Angular separation in lambda/D units
    """
    angle_rad = angle_arcsec * const.arcsec2rad
    wavelength_m = wavelength_nm * const.nm2m
    lambda_d_rad = wavelength_m / diameter_m
    return angle_rad / lambda_d_rad


def lambda_d_to_arcsec(angle_lambda_d, wavelength_nm, diameter_m):
    """Convert lambda/D units to angular separation.

    Args:
        angle_lambda_d: Angular separation in lambda/D units
        wavelength_nm: Wavelength in nanometers
        diameter_m: Telescope diameter in meters

    Returns:
        Angular separation in arcseconds
    """
    wavelength_m = wavelength_nm * const.nm2m
    lambda_d_rad = wavelength_m / diameter_m
    angle_rad = angle_lambda_d * lambda_d_rad
    return angle_rad * const.rad2arcsec


# Mass conversions
def Msun_to_kg(mass_solar: float) -> float:
    """Convert mass from solar masses to kilograms."""
    return mass_solar * const.Msun2kg


def Mearth_to_kg(mass_earth: float) -> float:
    """Convert mass from Earth masses to kilograms."""
    return mass_earth * const.Mearth2kg


# Time conversions
def years_to_days(time_years: jnp.ndarray) -> jnp.ndarray:
    """Convert time from years to days."""
    return time_years * 365.25


def days_to_years(time_days: jnp.ndarray) -> jnp.ndarray:
    """Convert time from days to years."""
    return time_days / 365.25


def decimal_year_to_jd(decimal_year: jnp.ndarray) -> jnp.ndarray:
    """Convert decimal year to Julian Date.

    Args:
        decimal_year: Year as a decimal (e.g., 2025.5)

    Returns:
        Julian Date
    """
    # JD for Jan 1, 2000 00:00 UT
    jd_2000 = 2451545.0

    # Days since 2000.0
    days_since_2000 = (decimal_year - 2000.0) * 365.25

    return jd_2000 + days_since_2000


# Distance conversions
def au_to_arcsec(distance_au: jnp.ndarray, distance_pc: float) -> jnp.ndarray:
    """Convert distance in AU to angular separation in arcseconds.

    Args:
        distance_au: Physical distance in AU
        distance_pc: Distance to system in parsecs

    Returns:
        Angular separation in arcseconds
    """
    # Small angle approximation: θ [arcsec] = distance [AU] / distance [parsec]
    return distance_au / distance_pc


def arcsec_to_au(angle_arcsec: jnp.ndarray, distance_pc: float) -> jnp.ndarray:
    """Convert angular separation to physical distance.

    Args:
        angle_arcsec: Angular separation in arcseconds
        distance_pc: Distance to system in parsecs

    Returns:
        Physical distance in AU
    """
    return angle_arcsec * distance_pc


def is_leap_year(year: float | jnp.ndarray) -> bool | jnp.ndarray:
    """Determine if a year is a leap year.

    Args:
        year:
            The year to check.

    Returns:
            True if the year is a leap year, False otherwise.
    """
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def days_in_year(year: float | jnp.ndarray) -> float | jnp.ndarray:
    """Return the number of days in a year.

    Args:
        year:
            The year to check.

    Returns:
            The number of days in the year (365 or 366).
    """
    return 365 + is_leap_year(year)


def gregorian_to_jd(year: float, month: float, day: float) -> float:
    """Convert a Gregorian date to a Julian day.

    This function calculates the Julian day for a given Gregorian date (year,
    month, day).

    Args:
        year:
            The year.
        month:
            The month.
        day:
            The day.

    Returns:
            The Julian day.
    """
    a = jnp.floor((14 - month) / 12)
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (
        day
        + jnp.floor((153 * m + 2) / 5)
        + 365 * y
        + jnp.floor(y / 4)
        - jnp.floor(y / 100)
        + jnp.floor(y / 400)
        - 32045
    )
    return jdn - 0.5


def jd_to_decimal_year(jd: float | jnp.ndarray) -> float | jnp.ndarray:
    """Convert a Julian day to a decimal year.

    Args:
        jd:
            The Julian day.

    Returns:
            The decimal year.
    """
    # Approximate year, good enough to find start of year
    year_approx = 1970.0 + (jd - 2440587.5) / 365.2425
    year = jnp.floor(year_approx)

    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    # Handle cases where the approximation was wrong
    # and jd is before the start of the calculated year
    year = jnp.where(jd < jd_start, year - 1, year)
    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    return year + (jd - jd_start) / (jd_end - jd_start)


def decimal_year_to_jd(
    decimal_year: float | jnp.ndarray,
) -> float | jnp.ndarray:
    """Convert a decimal year to a Julian day.

    Args:
        decimal_year:
            The decimal year.

    Returns:
            The Julian day.
    """
    year = jnp.floor(decimal_year)
    year_fraction = decimal_year - year

    jd_start = gregorian_to_jd(year, 1, 1)
    jd_end = gregorian_to_jd(year + 1, 1, 1)

    return jd_start + year_fraction * (jd_end - jd_start)
