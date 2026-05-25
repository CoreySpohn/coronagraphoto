"""Tests for simulation functions in coronagraphoto.simulation."""


class TestSimulationFunctions:
    """Tests for high-level simulation functions.

    Note: Full integration testing of simulation functions requires
    a complete optical path setup. These tests verify the interfaces
    are importable and callable.
    """

    def test_simulation_functions_exist(self):
        """Verify simulation functions are importable."""
        from coronagraphoto.simulation import (
            disk_rate,
            disk_readout,
            planet_rate,
            planet_readout,
            star_rate,
            star_readout,
            zodi_rate,
            zodi_readout,
        )

        # Verify they're callable
        assert callable(planet_rate)
        assert callable(star_rate)
        assert callable(disk_rate)
        assert callable(zodi_rate)
        assert callable(planet_readout)
        assert callable(star_readout)
        assert callable(disk_readout)
        assert callable(zodi_readout)

    def test_helper_functions_exist(self):
        """Verify helper functions are importable."""
        from coronagraphoto.simulation import (
            _convolve_quadrants,
            post_coro_bin_processing,
            pre_coro_bin_processing,
        )

        assert callable(_convolve_quadrants)
        assert callable(post_coro_bin_processing)
        assert callable(pre_coro_bin_processing)
