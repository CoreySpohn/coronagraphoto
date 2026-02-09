"""Tests for image and coordinate transforms in coronagraphoto.transforms."""

import jax
import jax.numpy as jnp
import pytest

from coronagraphoto.transforms.image_transforms import (
    ccw_rotation_matrix,
    resample_flux,
)
from coronagraphoto.transforms.orbital_mechanics import state_vector_to_keplerian


class TestRotationMatrix:
    """Tests for counter-clockwise rotation matrix generation."""

    def test_identity_at_zero(self):
        """0 degree rotation should give identity matrix."""
        rot_mat = ccw_rotation_matrix(0.0)
        identity = jnp.eye(2)
        assert jnp.allclose(rot_mat, identity, atol=1e-10)

    def test_90_degrees(self):
        """90 degree CCW rotation matrix values."""
        rot_mat = ccw_rotation_matrix(90.0)
        expected = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        assert jnp.allclose(rot_mat, expected, atol=1e-6)

    def test_180_degrees(self):
        """180 degree rotation should negate both axes."""
        rot_mat = ccw_rotation_matrix(180.0)
        expected = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        assert jnp.allclose(rot_mat, expected, atol=1e-6)

    def test_270_degrees(self):
        """270 degree CCW rotation (90 CW)."""
        rot_mat = ccw_rotation_matrix(270.0)
        expected = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        assert jnp.allclose(rot_mat, expected, atol=1e-6)

    def test_determinant_unity(self):
        """Rotation matrix should have determinant = 1."""
        for angle in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0]:
            rot_mat = ccw_rotation_matrix(angle)
            det = jnp.linalg.det(rot_mat)
            assert jnp.isclose(det, 1.0, atol=1e-10)


class TestResampleFlux:
    """Tests for flux-conserving image resampling."""

    def test_conserves_total_flux(self):
        """Total flux should be conserved during resampling."""
        # Create a simple test image with known total flux
        src_shape = (64, 64)
        tgt_shape = (32, 32)
        total_flux = 10000.0

        # Point source at center
        f_src = jnp.zeros(src_shape)
        f_src = f_src.at[32, 32].set(total_flux)

        # Resample to different pixel scale
        f_tgt = resample_flux(
            f_src,
            pixscale_src=0.01,
            pixscale_tgt=0.02,  # 2x larger pixels
            shape_tgt=tgt_shape,
            rotation_deg=0.0,
        )

        # Check flux conservation (should be close, not exact due to interpolation)
        assert jnp.isclose(jnp.sum(f_tgt), total_flux, rtol=0.1)

    def test_same_scale_identity(self):
        """Same pixel scale and shape should preserve image."""
        shape = (64, 64)
        f_src = jnp.ones(shape) * 100.0

        f_tgt = resample_flux(
            f_src,
            pixscale_src=0.01,
            pixscale_tgt=0.01,
            shape_tgt=shape,
            rotation_deg=0.0,
        )

        # Should be approximately equal
        assert jnp.allclose(f_tgt, f_src, rtol=0.01)

    def test_downsampling_flux_conservation(self):
        """Downsampling (larger pixels) should conserve flux."""
        f_src = jnp.ones((64, 64)) * 1.0  # 64x64 = 4096 total flux

        f_tgt = resample_flux(
            f_src,
            pixscale_src=0.01,
            pixscale_tgt=0.02,  # 2x larger pixels
            shape_tgt=(32, 32),
            rotation_deg=0.0,
        )

        # Total flux should be conserved
        assert jnp.isclose(jnp.sum(f_tgt), jnp.sum(f_src), rtol=0.05)

    def test_with_rotation(self):
        """Rotation should preserve most flux (some edge losses expected)."""
        f_src = jnp.ones((64, 64)) * 100.0
        total_flux_src = jnp.sum(f_src)

        f_tgt = resample_flux(
            f_src,
            pixscale_src=0.01,
            pixscale_tgt=0.01,
            shape_tgt=(64, 64),
            rotation_deg=45.0,
        )

        # Rotation causes some flux to fall outside image bounds
        # At 45°, corners are clipped so expect ~70-90% flux conservation
        ratio = jnp.sum(f_tgt) / total_flux_src
        assert 0.6 < ratio < 1.0, f"Flux ratio={ratio:.2f}, expected 0.7-1.0"


class TestStateVectorToKeplerian:
    """Tests for orbital mechanics conversion."""

    def test_circular_orbit(self):
        """Circular orbit should have e ≈ 0."""
        # 1 AU circular orbit around Sun
        mu = 1.327e20  # G * M_sun in m^3/s^2
        r_au = 1.0
        r_m = r_au * 1.496e11

        # Position on x-axis, velocity on y-axis (circular)
        r = jnp.array([r_m, 0.0, 0.0])
        v_circular = jnp.sqrt(mu / r_m)
        v = jnp.array([0.0, v_circular, 0.0])

        a, e, i, W, w, M = state_vector_to_keplerian(r, v, mu)

        # Check semi-major axis matches radius for circular orbit
        assert jnp.isclose(a, r_m, rtol=0.01)
        # Eccentricity should be near zero
        assert e < 0.01

    def test_elliptical_orbit(self):
        """Elliptical orbit should have e = 0.5."""
        mu = 1.327e20
        r_perihelion = 1.0 * 1.496e11  # 1 AU perihelion
        e_target = 0.5
        a_target = r_perihelion / (1 - e_target)

        # At perihelion, all velocity is tangential
        v_perihelion = jnp.sqrt(mu * (2 / r_perihelion - 1 / a_target))

        r = jnp.array([r_perihelion, 0.0, 0.0])
        v = jnp.array([0.0, v_perihelion, 0.0])

        a, e, i, W, w, M = state_vector_to_keplerian(r, v, mu)

        # Check we recover the expected eccentricity
        assert jnp.isclose(e, e_target, rtol=0.01)
        # Semi-major axis should match
        assert jnp.isclose(a, a_target, rtol=0.01)

    def test_inclined_orbit(self):
        """Inclined orbit should have i > 0."""
        mu = 1.327e20
        r_m = 1.496e11

        # Inclined at 30 degrees - position in x-z plane
        r = jnp.array([r_m * jnp.cos(jnp.pi / 6), 0.0, r_m * jnp.sin(jnp.pi / 6)])
        v_circular = jnp.sqrt(mu / r_m)
        # Velocity perpendicular to position
        v = jnp.array([0.0, v_circular, 0.0])

        a, e, i, W, w, M = state_vector_to_keplerian(r, v, mu)

        # Inclination should be non-zero
        assert i > 0.1  # More than ~5 degrees
