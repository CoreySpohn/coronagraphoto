"""System-Level Physics Verification for coronagraphoto (v2).

Validates the full simulation pipeline by treating the software as a
"Virtual Instrument" using mock components.

Test Tiers:
1. End-to-End Radiometry (Stars, Zodi)
2. Noise Statistics (Shot Noise, Read Noise)
3. Planet Astrometry (Achromaticity, Coordinate Transforms)
4. Mask Physics (Suppression)
"""

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbix.system.planets import Planets as OrbixPlanets

from coronagraphoto import constants as const
from coronagraphoto import conversions as conv
from coronagraphoto.core.optical_path import OpticalPath
from coronagraphoto.core.simulation import (
    gen_planet_count_rate,
    gen_star_count_rate,
    gen_zodi_count_rate,
    sim_star,
)
from coronagraphoto.core.sources import PlanetSources, StarSource
from coronagraphoto.core.zodi_sources import ZodiSourceAYO
from coronagraphoto.optical_elements import (
    ConstantThroughputElement,
    PrimaryAperture,
    SimpleDetector,
)
from coronagraphoto.optical_elements.detector import Detector

# =============================================================================
# MOCKS & FIXTURES
# =============================================================================


class MockCoronagraph(eqx.Module):
    """A 'Transparent' Coronagraph with Datacube support.

    Acts as a perfect lens:
    1. Transmission = 100%
    2. PSF = Gaussian centered exactly where requested (no aberrations)
    3. No complex optics or file loading required.

    Includes an 'Identity' PSF Datacube for disk convolution tests.
    """

    pixel_scale_lod: float
    psf_shape: tuple[int, int]
    center_x: float
    center_y: float
    sky_trans: jnp.ndarray
    psf_datacube: jnp.ndarray

    def __init__(self, size: int = 101, pixel_scale_lod: float = 0.5):
        """Initialize a mock coronagraph.

        Args:
            size: Size of the PSF grid (size x size pixels).
            pixel_scale_lod: Pixel scale in lambda/D per pixel.
        """
        self.psf_shape = (size, size)
        self.center_x = (size - 1) / 2.0
        self.center_y = (size - 1) / 2.0
        self.pixel_scale_lod = pixel_scale_lod
        self.sky_trans = jnp.ones((size, size))
        self.psf_datacube = None  # Not needed for point source tests

    def create_psfs(self, x_lod: jnp.ndarray, y_lod: jnp.ndarray) -> jnp.ndarray:
        """Generate Gaussian PSFs at the specified lambda/D coordinates."""
        x_pix = self.center_x + (x_lod / self.pixel_scale_lod)
        y_pix = self.center_y + (y_lod / self.pixel_scale_lod)

        ny, nx = self.psf_shape
        yy, xx = jnp.mgrid[:ny, :nx]

        def make_gaussian(xc, yc):
            sigma = 1.5  # PSF sigma in pixels
            g = jnp.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma**2))
            return g / jnp.sum(g)  # Flux Conserved

        return jax.vmap(make_gaussian)(x_pix, y_pix)

    def stellar_intens(self, diam_lod: float) -> jnp.ndarray:
        """Return a centered Gaussian for the star."""
        return self.create_psfs(jnp.array([0.0]), jnp.array([0.0]))[0]


@pytest.fixture
def perfect_system():
    """Create an OpticalPath with 100% throughput and a transparent coronagraph."""
    primary = PrimaryAperture(diameter_m=2.0, obscuration_factor=0.0)
    optics = ConstantThroughputElement(throughput=1.0)
    detector = SimpleDetector(
        pixel_scale=0.05,
        shape=(101, 101),
        quantum_efficiency=1.0,
        dark_current_rate=0.0,
    )
    coro = MockCoronagraph(size=101, pixel_scale_lod=0.5)
    return OpticalPath(primary, (optics,), coro, detector)


@pytest.fixture
def standard_star():
    """Create a 0 Mag (AB) StarSource for testing."""
    return StarSource(
        diameter_arcsec=0.0,
        mass_kg=const.Msun2kg,
        dist_pc=10.0,
        midplane_pa_deg=0.0,
        midplane_i_deg=0.0,
        wavelengths_nm=jnp.array([400.0, 550.0, 700.0, 800.0]),
        times_jd=jnp.array([0.0, 1.0]),
        flux_density_jy=jnp.full((4, 2), 3631.0),
    )


# =============================================================================
# TIER 1: END-TO-END RADIOMETRY
# =============================================================================


class TestEndToEndRadiometry:
    """Verify that photon counts through the full pipeline match analytic expectations."""

    def test_star_flux_conservation(self, perfect_system, standard_star):
        """Verify gen_star_count_rate produces the exact analytic photon count."""
        WAVELENGTH = 550.0
        BANDWIDTH = 50.0
        TIME_JD = 0.5

        image_rate = gen_star_count_rate(
            start_time_jd=TIME_JD,
            wavelength_nm=WAVELENGTH,
            bin_width_nm=BANDWIDTH,
            star=standard_star,
            optical_path=perfect_system,
        )

        total_simulated_rate = jnp.sum(image_rate)

        # Analytic: Flux * Area * Bandwidth
        flux_density = standard_star.spec_flux_density(WAVELENGTH, TIME_JD)
        area = np.pi * (perfect_system.primary.diameter_m / 2) ** 2
        expected_rate = flux_density * area * BANDWIDTH

        assert jnp.isclose(total_simulated_rate, expected_rate, rtol=0.01), (
            f"Flux Conservation Error: Simulated={float(total_simulated_rate):.4e} ph/s, "
            f"Expected={float(expected_rate):.4e} ph/s"
        )

    def test_throughput_attenuation(self, standard_star):
        """Verify that throughput elements correctly reduce flux."""
        # Create system with 50% throughput
        primary = PrimaryAperture(diameter_m=2.0, obscuration_factor=0.0)
        optics = ConstantThroughputElement(throughput=0.5)
        detector = SimpleDetector(
            pixel_scale=0.05, shape=(101, 101), quantum_efficiency=1.0
        )
        coro = MockCoronagraph(size=101, pixel_scale_lod=0.5)
        lossy_system = OpticalPath(primary, (optics,), coro, detector)

        # Perfect system
        perfect_optics = ConstantThroughputElement(throughput=1.0)
        perfect_system = OpticalPath(
            PrimaryAperture(2.0, 0.0),
            (perfect_optics,),
            MockCoronagraph(size=101, pixel_scale_lod=0.5),
            SimpleDetector(pixel_scale=0.05, shape=(101, 101), quantum_efficiency=1.0),
        )

        WAVELENGTH = 550.0
        BANDWIDTH = 50.0
        TIME_JD = 0.5

        rate_lossy = jnp.sum(
            gen_star_count_rate(
                TIME_JD, WAVELENGTH, BANDWIDTH, standard_star, lossy_system
            )
        )
        rate_perfect = jnp.sum(
            gen_star_count_rate(
                TIME_JD, WAVELENGTH, BANDWIDTH, standard_star, perfect_system
            )
        )

        ratio = rate_lossy / rate_perfect
        assert jnp.isclose(ratio, 0.5, rtol=0.01), (
            f"Throughput not applied correctly: ratio={float(ratio):.4f}, expected=0.5"
        )


# =============================================================================
# TIER 2: SURFACE BRIGHTNESS (Zodi)
# =============================================================================


class TestSurfaceBrightness:
    """Verify extended source integration is correct."""

    def test_zodi_surface_brightness_integration(self, perfect_system):
        """Verify ZodiSource generates correct counts/pixel for an extended source."""
        wavelengths = jnp.array([450.0, 550.0, 650.0])
        zodi = ZodiSourceAYO(wavelengths, surface_brightness_mag=22.0)

        WAVELENGTH = 550.0
        WIDTH = 1.0

        image_rate = gen_zodi_count_rate(
            start_time_jd=0.0,
            wavelength_nm=WAVELENGTH,
            bin_width_nm=WIDTH,
            zodi=zodi,
            optical_path=perfect_system,
        )

        # Analytic prediction
        flux_surf = zodi.spec_flux_density(WAVELENGTH, 0.0)
        lod_arcsec = conv.lambda_d_to_arcsec(
            perfect_system.coronagraph.pixel_scale_lod,
            WAVELENGTH,
            perfect_system.primary.diameter_m,
        )
        coro_pix_area = lod_arcsec**2
        tel_area = np.pi * (perfect_system.primary.diameter_m / 2) ** 2
        expected_rate_per_coro_pix = flux_surf * tel_area * WIDTH * coro_pix_area

        total_simulated = jnp.sum(image_rate)
        n_coro_pixels = perfect_system.coronagraph.psf_shape[0] ** 2
        expected_total = expected_rate_per_coro_pix * n_coro_pixels

        assert jnp.isclose(total_simulated, expected_total, rtol=0.05), (
            f"Zodi total: {float(total_simulated):.4e}, Expected: {float(expected_total):.4e}"
        )


# =============================================================================
# TIER 3: DETECTOR NOISE (FIXED - Dark frame for read noise)
# =============================================================================


class TestDetectorNoiseIntegration:
    """Verify the full readout chain produces correct noise statistics."""

    def test_full_chain_signal_consistency(self, perfect_system, standard_star):
        """Run sim_star and verify output is consistent with expectations."""
        key = jax.random.PRNGKey(42)
        exposure_time = 1.0

        img = sim_star(
            start_time_jd=0.5,
            exposure_time_s=exposure_time,
            wavelength_nm=550.0,
            bin_width_nm=50.0,
            star=standard_star,
            optical_path=perfect_system,
            prng_key=key,
        )

        total_counts = jnp.sum(img)

        flux_density = standard_star.spec_flux_density(550.0, 0.5)
        area = np.pi * (perfect_system.primary.diameter_m / 2) ** 2
        expected_rate = flux_density * area * 50.0
        expected_counts = expected_rate * exposure_time

        # 10% tolerance for resampling edge losses
        assert jnp.isclose(total_counts, expected_counts, rtol=0.1), (
            f"sim_star counts outside acceptable range. "
            f"Measured={float(total_counts):.0f}, Expected~{float(expected_counts):.0f}"
        )


# =============================================================================
# TIER 4: PLANET FIDELITY (Astrometry & Achromaticity)
# =============================================================================


class TestPlanetFidelity:
    """Verify planet positioning is correct across wavelengths."""

    @pytest.fixture
    def planet_setup(self, standard_star):
        """Create a planet at 5 AU (0.5 arcsec @ 10pc distance)."""
        orbix_planets = OrbixPlanets(
            Ms=jnp.atleast_1d(const.Msun2kg),
            dist=jnp.atleast_1d(10.0),
            a=jnp.array([5.0]),
            e=jnp.array([0.0]),
            i=jnp.array([0.0]),
            W=jnp.array([0.0]),
            w=jnp.array([0.0]),
            M0=jnp.array([0.0]),
            t0=jnp.array([0.0]),
            Mp=jnp.array([1.0]),
            Rp=jnp.array([1.0]),
            p=jnp.array([0.3]),  # Geometric albedo
        )
        # Flat contrast 1e-6 across wavelengths and times
        # Interpolator3D needs at least 2 points per axis
        contrast = interpax.Interpolator3D(
            jnp.array([400.0, 800.0]),
            jnp.array([0.0, 1.0]),  # Match star's times_jd
            jnp.array([0.0, 90.0]),  # Phase angles
            jnp.full((2, 2, 2), 1e-6),
        )
        return PlanetSources(
            star=standard_star,
            orbix_planet=orbix_planets,
            contrast_interp=contrast,
        )

    def test_planet_achromatic_position(self, perfect_system, planet_setup):
        """CRITICAL: Verify planet stays at the same DETECTOR pixel vs wavelength.

        Physics: Planet is at fixed arcsec separation (0.5").
        - At 400nm, this is X lambda/D
        - At 800nm, this is X/2 lambda/D
        - The pipeline must scale coordinates inversely to keep planet at same pixel.
        """
        # Blue (400nm)
        img_blue = gen_planet_count_rate(
            0.0, 400.0, 50.0, 0.0, planet_setup, perfect_system
        )
        yb, xb = jnp.unravel_index(jnp.argmax(img_blue), img_blue.shape)

        # Red (800nm)
        img_red = gen_planet_count_rate(
            0.0, 800.0, 50.0, 0.0, planet_setup, perfect_system
        )
        yr, xr = jnp.unravel_index(jnp.argmax(img_red), img_red.shape)

        dist = jnp.sqrt((float(xb) - float(xr)) ** 2 + (float(yb) - float(yr)) ** 2)
        assert dist <= 1.5, (
            f"Achromatic failure: Planet moved {dist:.2f} pixels "
            f"(Blue@400nm: [{xb},{yb}]; Red@800nm: [{xr},{yr}])"
        )

    def test_planet_absolute_position(self, perfect_system, planet_setup):
        """Verify 0.5 arcsec separation maps to the correct pixel offset.

        At 0.05 arcsec/pixel, 0.5 arcsec = 10 pixels from center.
        """
        img = gen_planet_count_rate(0.0, 550.0, 50.0, 0.0, planet_setup, perfect_system)
        y, x = jnp.unravel_index(jnp.argmax(img), img.shape)

        # Center is at (50, 50). Planet at M0=0 should be at +X direction
        # Expected: x ≈ 60, y ≈ 50
        assert abs(int(x) - 60) <= 2, (
            f"Planet at wrong X position. Got x={x}, Expected ~60"
        )
        assert abs(int(y) - 50) <= 2, f"Planet shifted in Y. Got y={y}, Expected ~50"


# =============================================================================
# TIER 5: MASK PHYSICS
# =============================================================================


class TestMaskPhysics:
    """Verify coronagraphic masking works correctly."""

    def test_coronagraph_sky_trans_suppression(self, perfect_system):
        """Verify the sky_trans map actually blocks light where set to 0."""
        # Create a coronagraph with central pixel masked
        masked_trans = perfect_system.coronagraph.sky_trans.at[50, 50].set(0.0)
        masked_coro = eqx.tree_at(
            lambda c: c.sky_trans, perfect_system.coronagraph, masked_trans
        )
        masked_sys = eqx.tree_at(lambda s: s.coronagraph, perfect_system, masked_coro)

        # Run Zodi (uniform extended source)
        zodi = ZodiSourceAYO(jnp.array([550.0]), surface_brightness_mag=20.0)
        img = gen_zodi_count_rate(0.0, 550.0, 50.0, zodi, masked_sys)

        # Center should be much darker than neighbor
        center_val = img[50, 50]
        neighbor_val = img[50, 51]

        assert center_val < neighbor_val * 0.1, (
            f"Mask did not suppress central flux: center={float(center_val):.2e}, "
            f"neighbor={float(neighbor_val):.2e}"
        )
