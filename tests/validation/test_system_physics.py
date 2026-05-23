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
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils import constants as const
from hwoutils import conversions as conv
from skyscapes.background import ZodiSourceAYO
from skyscapes.scene import SpectrumStar as StarSource

from coronagraphoto.core.optical_path import OpticalPath
from coronagraphoto.core.simulation import (
    gen_planet_count_rate,
    gen_star_count_rate,
    gen_zodi_count_rate,
    sim_star,
)
from coronagraphoto.optical_elements import (
    ConstantThroughputElement,
    PrimaryAperture,
    SimpleDetector,
)

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
    """Create a 0 Mag (AB) StarSource (SpectrumStar) for testing."""
    return StarSource(
        Ms_kg=const.Msun2kg,
        dist_pc=10.0,
        wavelengths_nm=jnp.array([400.0, 550.0, 700.0, 800.0]),
        times_jd=jnp.array([0.0, 1.0]),
        flux_density_jy=jnp.full((4, 2), 3631.0),
        diameter_arcsec=0.0,
    )


# =============================================================================
# TIER 1: END-TO-END RADIOMETRY
# =============================================================================


class TestEndToEndRadiometry:
    """Verify photon counts through the full pipeline.

    Checks that counts match analytic expectations.
    """

    def test_star_flux_conservation(self, perfect_system, standard_star):
        """Verify gen_star_count_rate produces the exact analytic photon count."""
        WAVELENGTH = 550.0
        BANDWIDTH = 50.0
        TIME_JD = 0.5

        image_rate = gen_star_count_rate(
            standard_star,
            perfect_system,
            start_time_jd=TIME_JD,
            wavelength_nm=WAVELENGTH,
            bin_width_nm=BANDWIDTH,
        )

        total_simulated_rate = jnp.sum(image_rate)

        # Analytic: Flux * Area * Bandwidth
        flux_density = standard_star.spec_flux_density(WAVELENGTH, TIME_JD)
        area = np.pi * (perfect_system.primary.diameter_m / 2) ** 2
        expected_rate = flux_density * area * BANDWIDTH

        assert jnp.isclose(total_simulated_rate, expected_rate, rtol=0.01), (
            "Flux Conservation Error: "
            f"Simulated={float(total_simulated_rate):.4e} ph/s, "
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
                standard_star,
                lossy_system,
                start_time_jd=TIME_JD,
                wavelength_nm=WAVELENGTH,
                bin_width_nm=BANDWIDTH,
            )
        )
        rate_perfect = jnp.sum(
            gen_star_count_rate(
                standard_star,
                perfect_system,
                start_time_jd=TIME_JD,
                wavelength_nm=WAVELENGTH,
                bin_width_nm=BANDWIDTH,
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
            zodi,
            perfect_system,
            start_time_jd=0.0,
            wavelength_nm=WAVELENGTH,
            bin_width_nm=WIDTH,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=135.0,
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
            f"Zodi total: {float(total_simulated):.4e}, "
            f"Expected: {float(expected_total):.4e}"
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
            standard_star,
            perfect_system,
            key,
            start_time_jd=0.5,
            exposure_time_s=exposure_time,
            wavelength_nm=550.0,
            bin_width_nm=50.0,
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
    def planet_setup(self):
        """Build a scene.Planet at 5 AU (0.5 arcsec @ 10pc) with grey contrast."""
        from orbix.kepler.shortcuts.grid import get_grid_solver
        from orbix.system.orbit import KeplerianOrbit
        from skyscapes.atmosphere import LambertianAtmosphere
        from skyscapes.scene import Planet, SpectrumStar

        star = SpectrumStar(
            Ms_kg=const.Msun2kg,
            dist_pc=10.0,
            wavelengths_nm=jnp.array([400.0, 550.0, 700.0, 800.0]),
            times_jd=jnp.array([0.0, 1.0]),
            flux_density_jy=jnp.full((4, 2), 3631.0),
            diameter_arcsec=0.0,
        )
        orbit = KeplerianOrbit(
            a_AU=jnp.array([5.0]),
            e=jnp.array([0.0]),
            W_rad=jnp.array([0.0]),
            i_rad=jnp.array([0.0]),
            w_rad=jnp.array([0.0]),
            M0_rad=jnp.array([0.0]),
            t0_d=jnp.array([0.0]),
        )
        # Grey atmosphere with Ag chosen so flux at M0=0 is non-trivial.
        atmosphere = LambertianAtmosphere(
            Rp_Rearth=jnp.array([1.0]),
            Ag=jnp.array([0.3]),
        )
        planet = Planet(orbit=orbit, atmosphere=atmosphere)
        solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
        return planet, star, solver

    def test_planet_achromatic_position(self, perfect_system, planet_setup):
        """Planet stays at the same DETECTOR pixel across wavelengths.

        Physics: planet is at a fixed arcsec separation. The pipeline must
        scale arcsec -> lambda/D correctly so the pixel location is
        wavelength-invariant.
        """
        planet, star, solver = planet_setup

        img_blue = gen_planet_count_rate(
            planet,
            perfect_system,
            start_time_jd=0.0,
            wavelength_nm=400.0,
            bin_width_nm=50.0,
            telescope_pa_deg=0.0,
            star=star,
            trig_solver=solver,
        )
        yb, xb = jnp.unravel_index(jnp.argmax(img_blue), img_blue.shape)

        img_red = gen_planet_count_rate(
            planet,
            perfect_system,
            start_time_jd=0.0,
            wavelength_nm=800.0,
            bin_width_nm=50.0,
            telescope_pa_deg=0.0,
            star=star,
            trig_solver=solver,
        )
        yr, xr = jnp.unravel_index(jnp.argmax(img_red), img_red.shape)

        dist = jnp.sqrt((float(xb) - float(xr)) ** 2 + (float(yb) - float(yr)) ** 2)
        assert dist <= 1.5, (
            f"Achromatic failure: planet moved {dist:.2f} pixels "
            f"(Blue@400nm: [{xb},{yb}]; Red@800nm: [{xr},{yr}])"
        )

    def test_planet_absolute_position(self, perfect_system, planet_setup):
        """0.5 arcsec separation maps to the correct pixel offset.

        At 0.05 arcsec/pixel, 0.5 arcsec = 10 pixels from center (50, 50).
        Planet at M0=0 lands at +X direction.
        """
        planet, star, solver = planet_setup
        img = gen_planet_count_rate(
            planet,
            perfect_system,
            start_time_jd=0.0,
            wavelength_nm=550.0,
            bin_width_nm=50.0,
            telescope_pa_deg=0.0,
            star=star,
            trig_solver=solver,
        )
        y, x = jnp.unravel_index(jnp.argmax(img), img.shape)
        assert abs(int(x) - 60) <= 2, (
            f"Planet at wrong X position. Got x={x}, expected ~60."
        )
        assert abs(int(y) - 50) <= 2, f"Planet shifted in Y. Got y={y}, expected ~50."

    def test_planet_count_rate_vmap_over_time(self, perfect_system):
        """Multi-time API is just ``jax.vmap`` over ``start_time_jd``.

        ``gen_planet_count_rate`` is single-time, but the function is
        pure JAX with no time-dependent control flow, so
        vmapping over ``start_time_jd`` returns a stack of per-epoch images
        with the orbit propagated independently at each time. Catches any
        future regression that would break this trace-time vectorisation
        (e.g. a Python branch on a traced time value).

        Uses a local ``SimpleStar`` instead of the ``planet_setup``
        fixture's ``SpectrumStar`` because the fixture's flux interpolant
        is built over ``times_jd=[0, 1]`` and extrapolates to NaN at the
        times we sweep here.
        """
        from orbix.kepler.shortcuts.grid import get_grid_solver
        from orbix.system.orbit import KeplerianOrbit
        from skyscapes.atmosphere import LambertianAtmosphere
        from skyscapes.scene import Planet, SimpleStar

        star = SimpleStar(
            Ms_kg=const.Msun2kg,
            dist_pc=10.0,
            flux_phot_per_nm_m2=1e9,
        )
        orbit = KeplerianOrbit(
            a_AU=jnp.array([5.0]),
            e=jnp.array([0.0]),
            W_rad=jnp.array([0.0]),
            i_rad=jnp.array([0.0]),
            w_rad=jnp.array([0.0]),
            M0_rad=jnp.array([0.0]),
            t0_d=jnp.array([0.0]),
        )
        atmosphere = LambertianAtmosphere(
            Rp_Rearth=jnp.array([1.0]),
            Ag=jnp.array([0.3]),
        )
        planet = Planet(orbit=orbit, atmosphere=atmosphere)
        solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)

        # Pick four times across roughly half an orbit. With a=5 AU around
        # a 1 Msun star, P ~ 11.18 yr ~ 4083 d, so 1500 d spans ~37% of an
        # orbit -- enough that the planet pixel position varies.
        times = jnp.array([0.0, 500.0, 1000.0, 1500.0])

        def at_time(t):
            return gen_planet_count_rate(
                planet,
                perfect_system,
                start_time_jd=t,
                wavelength_nm=550.0,
                bin_width_nm=50.0,
                telescope_pa_deg=0.0,
                star=star,
                trig_solver=solver,
            )

        images = jax.vmap(at_time)(times)

        # Shape: (T, det_ny, det_nx).
        assert images.shape == (len(times), *perfect_system.detector.shape)
        # All time slices finite and contain real flux.
        assert bool(jnp.all(jnp.isfinite(images)))
        assert bool(jnp.all(images.sum(axis=(1, 2)) > 0))

        # The brightest pixel should move between time slices, since the
        # planet is orbiting. If two consecutive epochs landed on the same
        # pixel something is wrong with the time threading.
        peaks = jnp.argmax(images.reshape(len(times), -1), axis=1)
        unique_peaks = len(set(int(p) for p in peaks))
        assert unique_peaks >= 2, (
            f"Planet pixel did not move across 4 epochs spanning ~1500 d "
            f"of a ~4080 d orbit; all peaks at {peaks.tolist()}."
        )


# =============================================================================
# TIER 5: MASK PHYSICS
# =============================================================================


class TestMaskPhysics:
    """Verify coronagraphic masking works correctly."""

    def test_sky_trans_modulates_zodi(self, perfect_system):
        """Verify sky_trans scales zodiacal background flux per pixel.

        sky_trans is the sky transmission map -- it modulates how much
        background (zodi) light reaches each coronagraph pixel. Setting
        it to zero in a region should zero out zodi flux there.
        """
        zodi = ZodiSourceAYO(jnp.array([550.0]), surface_brightness_mag=20.0)
        WAVELENGTH = 550.0
        BIN_WIDTH = 50.0

        zodi_kwargs = dict(
            start_time_jd=0.0,
            wavelength_nm=WAVELENGTH,
            bin_width_nm=BIN_WIDTH,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=135.0,
        )
        full_img = gen_zodi_count_rate(zodi, perfect_system, **zodi_kwargs)

        # Zero out a 5x5 patch in sky_trans
        masked_trans = perfect_system.coronagraph.sky_trans.at[48:53, 48:53].set(0.0)
        masked_coro = eqx.tree_at(
            lambda c: c.sky_trans,
            perfect_system.coronagraph,
            masked_trans,
        )
        masked_sys = eqx.tree_at(lambda s: s.coronagraph, perfect_system, masked_coro)

        masked_img = gen_zodi_count_rate(zodi, masked_sys, **zodi_kwargs)

        # Total flux should decrease
        assert jnp.sum(masked_img) < jnp.sum(full_img)


# =============================================================================
# TIER 6: ZODI DISPATCH (regression guard for nominal-union design)
# =============================================================================


class TestZodiTypeAgnosticDispatch:
    """Regression guard for nominal-union zodi dispatch.

    ``gen_zodi_count_rate`` must accept any concrete ``ZodiSource`` and
    route it through the same code path (no type-based branching).

    AYO and PhotonFlux share an identical photon-flux model when
    PhotonFlux is fed AYO's flux output, so their count-rate maps must
    match to machine precision. Leinert uses an independent
    position-and-wavelength model and is not expected to match
    numerically; we only assert it produces a finite, positive,
    same-shape map -- enough to prove dispatch works without making
    a false physical-equivalence claim.
    """

    def test_gen_zodi_count_rate_is_zodi_type_agnostic(self, perfect_system):
        """All three ZodiSource variants flow through gen_zodi_count_rate."""
        from skyscapes.background import (
            ZodiSourceLeinert,
            ZodiSourcePhotonFlux,
        )

        WAVELENGTH = 550.0
        BIN_WIDTH = 1.0
        wavelengths = jnp.array([WAVELENGTH])
        mag = 22.0

        ayo = ZodiSourceAYO(wavelengths, surface_brightness_mag=mag)
        phot = ZodiSourcePhotonFlux(
            wavelengths,
            jnp.array([ayo.spec_flux_density(WAVELENGTH, 0.0)]),
            reference_mag_arcsec2=mag,
        )
        leinert = ZodiSourceLeinert(reference_mag_arcsec2=mag)

        zodi_kwargs = dict(
            start_time_jd=0.0,
            wavelength_nm=WAVELENGTH,
            bin_width_nm=BIN_WIDTH,
            ecliptic_lat_deg=0.0,
            solar_lon_deg=135.0,
        )
        rate_ayo = gen_zodi_count_rate(ayo, perfect_system, **zodi_kwargs)
        rate_phot = gen_zodi_count_rate(phot, perfect_system, **zodi_kwargs)
        rate_leinert = gen_zodi_count_rate(leinert, perfect_system, **zodi_kwargs)

        assert rate_ayo.shape == rate_phot.shape == rate_leinert.shape

        # AYO and PhotonFlux are constructed to share the same flux at
        # WAVELENGTH; their per-pixel rates must match to machine precision.
        assert jnp.allclose(rate_ayo, rate_phot, rtol=1e-6, atol=0)

        # Leinert is a physically distinct model -- just assert it produces
        # a finite, positive map of the same shape (proves dispatch works).
        assert jnp.all(jnp.isfinite(rate_leinert))
        assert jnp.sum(rate_leinert) > 0


# =============================================================================
# TIER 7: DISK PIPELINE GUARDS
# =============================================================================


class TestDiskPipelineGuards:
    """Loud-fail guards on the disk simulation path."""

    def test_gen_disk_count_rate_raises_on_missing_psf_datacube(self):
        """A coronagraph without a PSF datacube must fail loudly, not silently."""
        from skyscapes.disk import ExovistaDisk
        from skyscapes.scene import SpectrumStar

        from coronagraphoto.core.simulation import gen_disk_count_rate

        class _CoroNoDatacube(eqx.Module):
            pixel_scale_lod: float = 0.5
            psf_shape: tuple = (51, 51)
            psf_datacube: object = None
            sky_trans: jnp.ndarray = eqx.field(
                default_factory=lambda: jnp.ones((51, 51))
            )

        primary = PrimaryAperture(diameter_m=2.0, obscuration_factor=0.0)
        optics = ConstantThroughputElement(throughput=1.0)
        detector = SimpleDetector(
            pixel_scale=0.05,
            shape=(51, 51),
            quantum_efficiency=1.0,
            dark_current_rate=0.0,
        )
        coro = _CoroNoDatacube()
        path = OpticalPath(primary, (optics,), coro, detector)

        wl = jnp.linspace(400.0, 700.0, 5)
        disk = ExovistaDisk(
            pixel_scale_arcsec=0.05,
            wavelengths_nm=wl,
            contrast_cube=jnp.full((wl.size, 51, 51), 1e-9),
        )
        star = SpectrumStar(
            Ms_kg=const.Msun2kg,
            dist_pc=10.0,
            wavelengths_nm=jnp.array([400.0, 550.0, 700.0]),
            times_jd=jnp.array([0.0, 1.0]),
            flux_density_jy=jnp.full((3, 2), 3631.0),
            diameter_arcsec=0.0,
        )

        with pytest.raises(ValueError, match="psf_datacube"):
            gen_disk_count_rate(
                disk,
                path,
                start_time_jd=0.0,
                wavelength_nm=550.0,
                bin_width_nm=50.0,
                telescope_pa_deg=0.0,
                star=star,
                incl_deg=0.0,
                pa_deg=0.0,
            )
