"""Tests for simulation functions in coronagraphoto.core.simulation."""

import jax.numpy as jnp
import pytest


class TestSimulationFunctions:
    """Tests for high-level simulation functions.

    Note: Full integration testing of simulation functions requires
    a complete optical path setup. These tests verify the interfaces
    are importable and callable.
    """

    def test_simulation_functions_exist(self):
        """Verify simulation functions are importable."""
        from coronagraphoto.core.simulation import (
            gen_disk_count_rate,
            gen_planet_count_rate,
            gen_star_count_rate,
            gen_zodi_count_rate,
            sim_disk,
            sim_planets,
            sim_star,
            sim_zodi,
        )

        # Verify they're callable
        assert callable(gen_planet_count_rate)
        assert callable(gen_star_count_rate)
        assert callable(gen_disk_count_rate)
        assert callable(gen_zodi_count_rate)
        assert callable(sim_planets)
        assert callable(sim_star)
        assert callable(sim_disk)
        assert callable(sim_zodi)

    def test_helper_functions_exist(self):
        """Verify helper functions are importable."""
        from coronagraphoto.core.simulation import (
            _convolve_quadrants,
            post_coro_bin_processing,
            pre_coro_bin_processing,
        )

        assert callable(_convolve_quadrants)
        assert callable(post_coro_bin_processing)
        assert callable(pre_coro_bin_processing)
