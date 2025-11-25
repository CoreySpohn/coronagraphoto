"""JAX friendly orbital mechanics functions."""

import jax.numpy as jnp


def state_vector_to_keplerian(r, v, mu):
    """Convert state vectors (r, v) to Keplerian elements using JAX.

    Robust implementation handling edge cases (circular, equatorial, and non-bound orbits)
    using jnp.where for JIT compatibility.

    Args:
        r (jnp.ndarray): Stellar-centric position vector (3,) in meters.
        v (jnp.ndarray): Stellar-centric velocity vector (3,) in m/s.
        mu (float): Gravitational parameter (G * M_total) in m^3/s^2.

    Returns:
        tuple: (a, e, i, W, w, M) - semi-major axis (m), eccentricity,
               inclination (rad), longitude of ascending node (W, rad),
               argument of periapsis (w, rad), mean anomaly (M, rad).
    """
    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    h = jnp.cross(r, v)
    h_mag = jnp.linalg.norm(h)

    i = jnp.arccos(jnp.clip(h[2] / h_mag, -1.0, 1.0))

    k = jnp.array([0.0, 0.0, 1.0])
    n = jnp.cross(k, h)
    n_mag = jnp.linalg.norm(n)

    e_vec = (1 / mu) * ((v_mag**2 - mu / r_mag) * r - jnp.dot(r, v) * v)
    e = jnp.linalg.norm(e_vec)

    E_energy = 0.5 * v_mag**2 - mu / r_mag
    a = jnp.where(jnp.abs(E_energy) > 1e-10, -mu / (2 * E_energy), jnp.inf)

    TOL_E = 1e-9
    TOL_I = 1e-9
    is_circular = e < TOL_E
    is_inclined = n_mag > TOL_I

    W = jnp.where(is_inclined, jnp.arctan2(n[1], n[0]), 0.0)

    # Inclined elliptical case
    cos_w = jnp.dot(n, e_vec) / (n_mag * e)
    w_inclined = jnp.arccos(jnp.clip(cos_w, -1.0, 1.0))
    w_inclined = jnp.where(e_vec[2] < 0, 2 * jnp.pi - w_inclined, w_inclined)

    # Equatorial elliptical case
    w_equatorial = jnp.arctan2(e_vec[1], e_vec[0])
    w_equatorial = w_equatorial * jnp.sign(h[2])

    w = jnp.where(is_circular, 0.0, jnp.where(is_inclined, w_inclined, w_equatorial))

    # True anomaly for elliptical orbits
    cos_nu = jnp.dot(e_vec, r) / (e * r_mag)
    nu_elliptical = jnp.arccos(jnp.clip(cos_nu, -1.0, 1.0))
    nu_elliptical = jnp.where(
        jnp.dot(r, v) < 0, 2 * jnp.pi - nu_elliptical, nu_elliptical
    )

    # Circular inclined case (argument of latitude)
    cos_u = jnp.dot(n, r) / (n_mag * r_mag)
    u_inclined = jnp.arccos(jnp.clip(cos_u, -1.0, 1.0))
    u_inclined = jnp.where(r[2] < 0, 2 * jnp.pi - u_inclined, u_inclined)

    # Circular equatorial case (true longitude)
    nu_equatorial = jnp.arctan2(r[1], r[0])
    nu_equatorial = nu_equatorial * jnp.sign(h[2])

    nu = jnp.where(
        is_circular, jnp.where(is_inclined, u_inclined, nu_equatorial), nu_elliptical
    )

    W = W % (2 * jnp.pi)
    w = w % (2 * jnp.pi)
    nu = nu % (2 * jnp.pi)

    E = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
    M = E - e * jnp.sin(E)
    M = M % (2 * jnp.pi)
    M = jnp.where(e < 1.0, M, jnp.nan)

    return a, e, i, W, w, M
