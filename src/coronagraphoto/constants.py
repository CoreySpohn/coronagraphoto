"""Physical constants and unit conversions for coronagraphoto."""

import jax.numpy as jnp

# Mathematical constants
two_pi = 2 * jnp.pi
pi_over_2 = jnp.pi / 2
eps = jnp.finfo(jnp.float32).eps

# Physical constants
h = 6.62607015e-34  # Planck constant in J⋅s
c = 299792458.0  # Speed of light in m/s
k_B = 1.380649e-23  # Boltzmann constant in J/K
sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant in W⋅m^-2⋅K^-4
G_kg_m_s = 6.67430e-11  # m^3 / (kg s^2)
Rearth2m = 6.371e6  # Earth radii to meters

# Flux conversion constants
Jy = 1e-26  # Jansky in W⋅m^-2⋅Hz^-1

# Length conversions
nm2m = 1e-9  # nanometers to meters
m2nm = 1e9  # meters to nanometers
um2m = 1e-6  # micrometers to meters
m2um = 1e6  # meters to micrometers
nm2um = 1e-3  # nanometers to micrometers
um2nm = 1e3  # micrometers to nanometers

# Angular conversions (same as orbix for consistency)
rad2arcsec = 206264.80624709636  # radians to arcseconds
arcsec2rad = 4.84813681109536e-06  # arcseconds to radians
mas2arcsec = 1e-3  # milliarcseconds to arcseconds
arcsec2mas = 1e3  # arcseconds to milliarcseconds
deg2rad = jnp.pi / 180.0  # degrees to radians
rad2deg = 180.0 / jnp.pi  # radians to degrees

# Time conversions
yr2s = 365.25 * 86400.0  # years to seconds
s2yr = 1.0 / yr2s  # seconds to years
d2s = 86400.0  # days to seconds (same as orbix)
s2d = 1.157407407407407e-05  # seconds to days (same as orbix)
J2000_JD = 2451545.0  # J2000 epoch in Julian days

# Distance conversions (from orbix for planet calculations)
AU2m = 1.495978707e11  # AU to meters
m2AU = 6.684587122268445e-12  # meters to AU
pc2m = 3.0857e16  # parsecs to meters
m2pc = 1.0 / pc2m  # meters to parsecs
pc2AU = 2.062648062470964e05  # parsecs to AU (same as orbix)

# Mass conversions (from orbix for planet calculations)
Msun2kg = 1.988409870698051e30  # solar masses to kg
Mearth2kg = 5.972167867791379e24  # Earth masses to kg
Mjup2kg = 1.898124571735094e27  # Jupiter masses to kg

# Detector-related constants
e_per_ph = 1.0  # electrons per photon (placeholder, depends on QE)
DN_per_e = 1.0  # digital numbers per electron (placeholder, depends on gain)
