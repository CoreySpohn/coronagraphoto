"""Shared pytest fixtures and data utilities for coronagraphoto tests."""

# Precision is configured before importing the numerical model classes.
# ruff: noqa: E402

import importlib.util
import os
import sys
from dataclasses import replace

sys.modules.setdefault("conftest", sys.modules[__name__])

import jax
import pytest
from hypothesis import settings

# Keep lazily cached dependency tables and evaluation arrays at one precision.
jax.config.update("jax_enable_x64", True)

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


@pytest.fixture(scope="module")
def exovista_precision():
    """Preserve precision when constructing imported orbital states."""
    with jax.enable_x64(True):
        yield


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prng_key():
    """Provide a reproducible JAX random key."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def wavelength_nm():
    """Standard test wavelength (V-band)."""
    return 550.0


@pytest.fixture
def exposure_time_s():
    """Standard test exposure time (1 hour)."""
    return 3600.0


@pytest.fixture
def detector_shape():
    """Standard detector shape for tests."""
    return (100, 100)


@pytest.fixture
def pixel_scale_arcsec():
    """Standard pixel scale in arcsec/pixel."""
    return 0.01  # 10 mas/pixel


# All small synthetic scene/instrument helpers live here for reuse.
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import zodi
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from jax.scipy.special import erf
from optixstuff import ConstantThroughput, Detector, Exposure, Instrument, SimplePrimary
from optixstuff.apertures import Circular
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    AbstractSceneCoronagraph,
    AbstractTableCoronagraph,
    ProductGrids,
    SceneRender,
    SourcePSFs,
)
from optixstuff.grids import Frame, Image, Measure, PixelGrid, resample
from optixstuff.optical_elements import TransmissiveElement
from optixstuff.performance import PerformanceCurves, SampledProfile
from optixstuff.spectral import Band, Resolution, Transmission
from orbix.observatory import ecliptic_pointing
from scipy.special import erf as scipy_erf
from skyscapes.physical_model import AbstractPhysicalModel, GridPhysicalModel
from skyscapes.scene import FlatStar, Planet, Scene, Star

_RULE = Circular(0.7)


def sightline(scene, time_d):
    """Ecliptic latitude and solar longitude of the scene at ``time_d``."""
    return ecliptic_pointing(scene.ra_deg, scene.dec_deg, scene.epoch, time_d)


def leinert_radiance(scene, wavelength_nm, time_d):
    """Leinert photon radiance of the scene's sightline, from zodi directly.

    An oracle independent of ``LeinertZodi``: the sightline angles come from
    ``ecliptic_pointing`` and the radiance from ``zodi.specific_intensity``.
    """
    latitude_deg, longitude_deg = sightline(scene, time_d)
    return (
        zodi.specific_intensity(
            longitude_deg,
            latitude_deg,
            jnp.asarray(wavelength_nm),
            photon_units=True,
            anchor="table19",
        )
        / 1000.0
    )


def with_zodi(scene, zodi):
    """The scene with its local zodi replaced."""
    return eqx.tree_at(lambda s: s.local_zodi, scene, zodi, is_leaf=lambda x: x is None)


class GaussianCoronagraph(AbstractTableCoronagraph):
    """Analytic flux-fraction Gaussian tables, sigma=.35 lod and tau=.3.

    No renormalization at image boundaries. The finite-field truncation is
    explicitly controlled by the fixture dimensions.
    """

    grid: PixelGrid = eqx.field(static=True)
    sky_trans: Image
    uniform_sky_profile: SampledProfile
    bandpass: Band | None = None
    sigma_lod: float = 0.35
    band_averaged: bool = eqx.field(static=True, default=False)
    chromatic: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        n=40,
        step=0.25,
        *,
        chromatic=False,
        frame=None,
        bandpass=None,
        band_averaged=False,
    ):
        """Build an n x n focal grid with the given lambda/D step."""
        if frame is None:
            frame = Frame(550.0, 6.0, "circumscribed")
        self.bandpass = bandpass
        self.band_averaged = band_averaged
        self.grid = PixelGrid.focal(n, step, frame)
        self.chromatic = chromatic
        self.sky_trans = Image(
            jnp.full((n, n), 0.3), self.grid, (), Measure.intensive(), "transmission"
        )
        self.uniform_sky_profile = SampledProfile(
            [0.0],
            [0.3],
            quantity="uniform_sky_response",
            frame=self.grid.frame,
            interpolation="linear",
        )

    def _native_source_psfs(self, x_lod, y_lod):
        """Integrate the sampled Gaussian with its stated native pixel area."""
        x, y = self.grid.coordinates()
        r2 = (x[None] - jnp.atleast_1d(x_lod)[:, None, None]) ** 2 + (
            y[None] - jnp.atleast_1d(y_lod)[:, None, None]
        ) ** 2
        data = (
            0.3
            * self.grid.pixel_area
            / (2 * np.pi * self.sigma_lod**2)
            * jnp.exp(-r2 / (2 * self.sigma_lod**2))
        )
        image = Image(
            data, self.grid, ("source",), Measure.integral(), "stellar_flux_fraction"
        )
        return SourcePSFs(image, jnp.ones(data.shape[0], dtype=bool))

    def _native_stellar_map(self, stellar_diameter_lod):
        """Uniform leakage density with a resolved-diameter response."""
        return Image(
            jnp.full(self.grid.shape, 1e-9 * self.grid.pixel_area)
            * (1 + stellar_diameter_lod**2),
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Optional chromatic throughput used by the worked example."""
        result = super().source_psfs(wavelength_nm, x_lod, y_lod, grid=grid)
        factor = 1 + 0.002 * (wavelength_nm - 550) if self.chromatic else 1.0
        return SourcePSFs(
            replace(result.image, data=result.image.data * factor), result.valid
        )

    def curves(self, rule=_RULE, *, leakage_rule="aperture_mean"):
        """Closed-form circular Gaussian and constant leakage integrals."""
        area = np.pi * rule.radius_lod**2
        tau = 0.3 * (1 - np.exp(-(rule.radius_lod**2) / (2 * self.sigma_lod**2)))
        extent = min(self.grid.shape) * self.grid.step / 2
        return PerformanceCurves(
            [0.0, extent],
            self.grid.frame,
            rule,
            [tau, tau],
            [area, area],
            [1e-9 * area] * 2,
            [1e-9] * 2,
            [1e-9 * area / tau] * 2,
            0.0,
            extent,
            throughput_variance=[tau, tau],
            core_area_variance_lod2=[area, area],
            leakage_fraction_variance=[1e-9 * area] * 2,
            leakage_density_variance_lod2=[1e-9] * 2,
        )


class _GaussianSceneOperator(eqx.Module):
    """One closed-form Gaussian PSF per source pixel, contracted with the flux.

    Satisfies the scene-operator contract of yippy (``render(flux, grid)``) for an
    analytic model that has no yield input package to build an operator from.
    """

    psfs: jax.Array
    valid: jax.Array

    def render(self, flux, grid):
        """Return the image and whether every lit source is valid."""
        valid = jnp.all((flux == 0) | (jnp.isfinite(flux) & self.valid))
        image = jnp.einsum("ij,ijyx->yx", flux, self.psfs)
        return jnp.where(valid, image, jnp.nan), valid


class GaussianSceneCoronagraph(GaussianCoronagraph, AbstractSceneCoronagraph):
    """Dense scene operator built from the same synthetic off-axis PSFs."""

    operator: object

    def __init__(self, n=40, step=0.25, *, chromatic=False):
        """Prepare a full set of native Gaussian source responses."""
        super().__init__(n, step, chromatic=chromatic)
        x, y = self.grid.coordinates()
        psfs = self.source_psfs(550.0, x.ravel(), y.ravel(), grid=None)
        self.operator = _GaussianSceneOperator(
            psfs.image.data.reshape(n, n, n, n), psfs.valid.reshape(n, n)
        )

    @property
    def scene_grid(self):
        """The dense operator reads the stored PSF grid."""
        return self.grid

    def render_scene(self, scene, wavelength_nm, *, grid):
        """Apply the dense operator to per-pixel incident flux."""
        flux = resample(scene, self.grid).data
        data, valid = self.operator.render(flux, self.grid)
        native = Image(
            data,
            self.grid,
            (),
            Measure.integral(),
            scene.quantity,
            reconstruction="band_limited",
        )
        return SceneRender(resample(native, grid), valid)


class FlatContrast(AbstractPhysicalModel):
    """A planet with one contrast at every wavelength and no knots.

    A knotless model needs no resolution, so a hand-made quadrature (nodes that
    are traced leaves, or a rule written by hand) can render it.
    """

    value: jnp.ndarray

    def contrast(
        self, phase_angle_rad, star_planet_distance_AU, wavelength_nm, Rp_Rearth
    ):
        """The same contrast for every planet, time and wavelength."""
        return jnp.broadcast_to(self.value, phase_angle_rad.shape)


def exposure_for(scene, quadrature, *, time_d=0.0, roll_deg=0.0, plan=None):
    """An exposure on a hand-built quadrature, with the scene's resolution.

    The quadrature must come from ``Quadrature.build`` when the scene has a
    tabulated source.
    """
    return Exposure(
        time_d=time_d,
        roll_deg=roll_deg,
        quadrature=quadrature,
        plan=plan,
        resolution=Resolution.build(quadrature, *scene.spectral_tables()),
    )


def make_system(
    *,
    star=None,
    contrast=1e-10,
    sky_pa_deg=0.0,
    separation_arcsec=0.05,
    disk=None,
    epoch=None,
    epoch_d=0.0,
    planet_groups=None,
    smooth_planet=False,
    wavelengths_nm=(400.0, 500.0, 600.0, 700.0),
    host=False,
):
    """Real static-spectrum photometry with a circular orbit at its node.

    The planet's contrast is tabulated on ``wavelengths_nm`` unless
    ``smooth_planet``, which uses a model with no knots. With ``host`` the
    planet's table, the default star's table (a short tabulated star in place of
    the flat one) and the orbit arrays are given as NumPy arrays, as a loader
    hands them over.
    """
    xp = np if host else jnp
    if star is None and host:
        star = Star(
            wavelengths_nm=np.array([400.0, 550.0, 700.0]),
            times_d=np.array([0.0]),
            flux_density_jy=np.array([[1.0], [2.0], [1.5]]),
        )
    star = FlatStar(flux_phot_per_nm_m2=1e6) if star is None else star
    wavelengths = xp.asarray(wavelengths_nm)
    contrast = xp.asarray(contrast)
    values = jnp.broadcast_to(contrast, (wavelengths.size,))
    if smooth_planet:
        model = FlatContrast(contrast)
    else:
        model = GridPhysicalModel(
            wavelengths,
            xp.array([0.0, 90.0, 180.0]),
            xp.broadcast_to(values[None, :, None], (1, wavelengths.size, 3)),
        )
    planet = Planet(jnp.ones(1), model)
    return Scene.from_classical(
        star,
        (planet,) if planet_groups is None else planet_groups,
        star_mass_Msun=1.0,
        planet_mass_Msun=Mearth2kg / Msun2kg,
        dist_pc=10.0,
        epoch=Epoch(2451545.0) if epoch is None else epoch,
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=xp.atleast_1d(separation_arcsec) * 10,
        e=xp.zeros(1),
        W_rad=xp.deg2rad(xp.asarray(sky_pa_deg)),
        i_rad=xp.zeros(1),
        w_rad=xp.zeros(1),
        M0_rad=xp.zeros(1),
        epoch_d=epoch_d,
        disk=disk,
    )


def make_path(
    *, n=40, step=0.25, detector_n=None, detector_step=None, chromatic=False, cube=False
):
    """40 m2 collecting area, .4 optical throughput, separate spectral yield."""
    model = GaussianSceneCoronagraph if cube else GaussianCoronagraph
    coro = model(n, step, chromatic=chromatic)
    scale = float(coro.grid.frame.lod_to_arcsec(550.0))
    detector = Detector(
        pixel_scale_arcsec=(step if detector_step is None else detector_step) * scale,
        shape=(n if detector_n is None else detector_n,) * 2,
        # Exact .8 - .002*(lambda-550) in the worked interval.
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
    )
    return Instrument(
        SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=40.0),
        (ConstantThroughput(0.4),),
        coro,
        detector,
    )


def worked_setup(*, n_wavelengths=1201):
    """AB=5 flat F_nu, dense cubic photon interpolation, linear contrast.

    The coronagraph tables reach 12 lambda/D (n=96 at 0.25) so that the 7.2
    lambda/D detector half-width is supported at the 400 nm node, where the
    tables' footprint on the sky is 400/550 of its width at 550 nm; nothing
    outside the tables is rendered.
    """
    wave = jnp.linspace(400.0, 700.0, n_wavelengths)
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.full((wave.size, 1), 3631.0 * 10 ** (-0.4 * 5)),
    )
    contrast = 1e-10 * (1 - 0.004 * (jnp.array([400.0, 500.0, 600.0, 700.0]) - 550))
    return make_system(star=star, contrast=contrast), make_path(
        n=96, chromatic=True, detector_n=48, detector_step=0.3
    )


def centroid(data, grid):
    """Centroid in pixel coordinates, retaining geometric even-grid centres."""
    yy, xx = np.indices(grid.shape)
    data = np.asarray(data)
    return np.array([(data * xx).sum(), (data * yy).sum()]) / data.sum()


class PlateauCoronagraph(GaussianSceneCoronagraph):
    """Manufactured square PSF: .3/16 per lod2 in a 4-by-4 lod square.

    Half-weight boundary samples give total .3 on the native source lattice.
    Circular(.7) lies entirely in its plateau, so tau=.3*pi*.7**2/16.
    This flat-aperture case separates radiometric closure from quadrature
    of curved PSFs, tested by the Gaussian worked example.
    """

    def _native_source_psfs(self, x_lod, y_lod):
        """Separable square with the trapezoidal convention on its boundary."""
        x, y = self.grid.coordinates()
        dx = jnp.abs(x[None] - jnp.atleast_1d(x_lod)[:, None, None])
        dy = jnp.abs(y[None] - jnp.atleast_1d(y_lod)[:, None, None])
        wx = jnp.where(dx < 2.0, 1.0, jnp.where(dx == 2.0, 0.5, 0.0))
        wy = jnp.where(dy < 2.0, 1.0, jnp.where(dy == 2.0, 0.5, 0.0))
        image = Image(
            0.3 / 16 * self.grid.pixel_area * wx * wy,
            self.grid,
            ("source",),
            Measure.integral(),
            "stellar_flux_fraction",
        )
        return SourcePSFs(image, jnp.ones(image.data.shape[0], dtype=bool))

    def curves(self, rule=_RULE, *, leakage_rule="aperture_mean"):
        """Independent exact area integrals of the manufactured uniform fields."""
        area = np.pi * rule.radius_lod**2
        tau, ell = 0.3 / 16 * area, 1e-9 * area
        extent = min(self.grid.shape) * self.grid.step / 2
        return PerformanceCurves(
            [0.0, extent],
            self.grid.frame,
            rule,
            [tau, tau],
            [area, area],
            [ell, ell],
            [1e-9] * 2,
            [ell / tau] * 2,
            0.0,
            extent,
            throughput_variance=[tau, tau],
            core_area_variance_lod2=[area, area],
            leakage_fraction_variance=[ell, ell],
            leakage_density_variance_lod2=[1e-9] * 2,
        )


def write_scene(path, *, a_au=1.0):
    """Small ExoVista v2.6 FITS with a native disk and independent sky states.

    The planet orbits at ``a_au`` astronomical units.
    """
    from astropy.io import fits
    from hwoutils.constants import AU2m, G_si, yr2s

    wave = np.array([0.5, 0.6, 0.7])
    primary = fits.PrimaryHDU(wave)
    primary.header["VERSION"] = 2.6
    primary.header["N_EXT"] = 6
    cube = np.full((4, 12, 12), 1e-10)
    disk = fits.ImageHDU(cube, header=fits.Header({"PXSCLMAS": 10.0}))
    star = np.zeros((2, 19))
    star[:, 0] = [0.0, 0.001]
    star[:, 16:] = 1.0
    header = fits.Header(
        {
            "MASS": 1.0,
            "DIST": 10.0,
            "RA": 100.0,
            "DEC": -20.0,
            "RSTAR": 1.0,
            "ANGDIAM": 0.9,
            "LSTAR": 1.0,
            "I": 60.0,
            "PA": 30.0,
            "TEFF": 5778.0,
            "SPECUNIT": "Jy",
        }
    )
    planet = np.zeros_like(star)
    planet[:, 0] = star[:, 0]
    # A circular orbit of radius a_au: the loader reads these states, not the header A
    planet[:, 9] = a_au
    planet[:, 13] = np.sqrt(G_si * (Msun2kg + Mearth2kg) / (a_au * AU2m)) * yr2s / AU2m
    planet[:, 16:] = 1e-10
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(wave),
            disk,
            fits.ImageHDU(),
            fits.ImageHDU(np.zeros((2, 0, 4))),
            fits.ImageHDU(star, header=header),
            fits.ImageHDU(
                planet, header=fits.Header({"R": 1.0, "M": 1.0, "A": a_au, "E": 0.0})
            ),
        ]
    ).writeto(path)
    return str(path)


def closure_system(scale):
    """The scene of the aperture closure tests: one planet, a disk and a local zodi.

    ``scale`` is the arcsecond size of one lambda/D at 550 nm. The planet sits
    at 2.3 lambda/D, contrast 1e-9, with a parametric disk and a uniform local
    zodi radiance.
    """
    from skyscapes import PrecomputedZodi
    from skyscapes.disk import GraterDisk
    from zodi import SurfaceBrightnessDust

    disk = GraterDisk(
        sma_AU=1.0,
        alpha_in=2.0,
        alpha_out=-2.0,
        ksi0_AU=0.05,
        gamma=2.0,
        beta=1.0,
        rmin_AU=0.1,
        rmax_AU=5.0,
        wavelengths_nm=jnp.array([500.0, 600.0]),
        g_HG_grid=jnp.zeros(2),
        Ag_grid=jnp.ones(2),
        amplitude=SurfaceBrightnessDust(level=3.0),
        n_slices_los=24,
    )
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    return with_zodi(
        make_system(
            contrast=1e-9, separation_arcsec=2.3 * scale, sky_pa_deg=250.0, disk=disk
        ),
        zodi,
    )


# Pixel pitch and Gaussian sigma of the synthetic YIP, in lambda/D.
YIP_PITCH_LOD, YIP_SIGMA_LOD = 0.5, 0.7


def write_yip(
    path, *, distinct_products=False, stellar_profile=None, reach=3, half_width=6
):
    """Single-wavelength Gaussian YIP with SI-labelled headers and even centre.

    ``stellar_profile`` is an optional function of the radius in lambda/D that
    multiplies the uniform stellar leakage, for a leakage that varies with
    separation. ``reach`` is the half-width in lambda/D of the off-axis grid (a
    square of nodes at 1 lambda/D) and ``half_width`` that of the image tables;
    a closure that reads apertures up to a separation ``s`` needs both above
    ``s`` plus the aperture radius, since nothing outside them is rendered.
    """
    from yippy.testing import write_synthetic_yip
    from yippy.writer import yip_header

    step, sigma = YIP_PITCH_LOD, YIP_SIGMA_LOD
    n = round(2 * half_width / step)
    yy, xx = np.indices((n, n))
    xx, yy = (xx - (n - 1) / 2) * step, (yy - (n - 1) / 2) * step
    offsets = np.array(
        [(x, y) for x in range(reach + 1) for y in range(reach + 1)], dtype=float
    )
    psfs = np.array(
        [
            0.3
            * step**2
            / (2 * np.pi * sigma**2)
            * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
            for x, y in offsets
        ]
    )
    grid = PixelGrid.focal(n, step, Frame(550.0, 6.0, "circumscribed"))
    diameters = np.array([0.0, 0.1, 0.2])
    stellar = np.full((len(diameters), n, n), 1e-9 * step**2)
    if stellar_profile is not None:
        stellar *= stellar_profile(np.hypot(xx, yy))
    sky = np.full((n, n), 0.3)
    metadata = dict(
        inscribed_diameter_m=4.7,
        obscured=0.2,
        low_nm=550.0,
        high_nm=550.0,
        n_wavelengths=1,
        n_pointings=3,
    )
    headers = yip_header(grid, **metadata)
    if distinct_products:
        stellar *= 1 + diameters[:, None, None] ** 2
        stellar *= 1 + 0.02 * xx
        sky += 0.01 * xx
        sky_grid = PixelGrid.focal(n, step, Frame(750.0, 4.0, "circumscribed"))
        headers = {
            "offaxis_psfs": headers,
            "offsets_xy_lod": headers,
            "stellar_intensity": headers,
            "stellar_diameters_lod": headers,
            "sky_transmission": yip_header(sky_grid, **metadata),
        }
    write_synthetic_yip(
        path,
        grid=grid,
        offsets_xy_lod=offsets,
        psfs=psfs,
        stellar=(diameters, stellar),
        sky=sky,
        headers=headers,
    )
    return str(path)


class IntegratedGaussianCoronagraph(AbstractImagingCoronagraph):
    """Two resolved stellar lobes and a Gaussian planet, integrated on demand."""

    bandpass: Band = eqx.field(static=True)
    band_averaged: bool = eqx.field(static=True)
    breaks_nm: tuple = eqx.field(static=True, default=())
    throughput: float = 0.3
    grid: PixelGrid = eqx.field(
        static=True,
        default=PixelGrid.focal(32, 0.4, Frame(600.0, 6.0, "circumscribed")),
    )

    @property
    def frame(self):
        """The reference frame shared by all three products."""
        return self.grid.frame

    def spectral_breaks_nm(self):
        """Include the optional internal breaks in the calibrated interval."""
        return np.union1d(super().spectral_breaks_nm(), self.breaks_nm)

    def at_wavelength(self, wavelength_nm):
        """All products cover the same square in lambda/D."""
        return ProductGrids(self.grid, self.grid, self.grid)

    @staticmethod
    def _gaussian(grid, cx, cy, sigma):
        """Unit-flux Gaussian integrated over the axis-aligned pixel rectangles."""
        x, y = grid.coordinates()

        def axis(delta, pitch):
            width = jnp.sqrt(2.0) * sigma
            return (
                erf((delta + pitch / 2) / width) - erf((delta - pitch / 2) / width)
            ) / 2

        return axis(x - cx, grid.basis[0][0]) * axis(y - cy, grid.basis[1][1])

    def stellar_map(self, wavelength_nm, stellar_diameter_lod, *, grid=None):
        """Leakage lobes broaden in quadrature with stellar diameter."""
        grid = self.grid if grid is None else grid
        width = stellar_diameter_lod / 2
        data = self._gaussian(grid, -0.6, 0.45, jnp.hypot(0.28, width))
        data += 0.6 * self._gaussian(grid, 0.8, -0.35, jnp.hypot(0.42, width))
        return Image(1e-8 * data, grid, (), Measure.integral(), "stellar_flux_fraction")

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Integrate one off-axis Gaussian per source."""
        grid = self.grid if grid is None else grid
        x, y = (jnp.atleast_1d(v)[:, None, None] for v in (x_lod, y_lod))
        data = self.throughput * self._gaussian(grid, x, y, 0.35)
        image = Image(
            data, grid, ("source",), Measure.integral(), "stellar_flux_fraction"
        )
        return SourcePSFs(image, jnp.ones(x.shape[0], dtype=bool))

    def uniform_sky_map(self, wavelength_nm, *, grid):
        """Constant dimensionless transmission."""
        return Image(
            jnp.full(grid.shape, 0.3), grid, (), Measure.intensive(), "transmission"
        )


@pytest.fixture
def gaussian_observation():
    """Build a 540 nm observation with a resolved star and sloping gain and QE."""
    from coronagraphoto import build_exposure

    def build(band_averaged=True, *, coro=None):
        band = Band(520, 560)
        if coro is None:
            coro = IntegratedGaussianCoronagraph(
                Band(480, 720), band_averaged, (543.0,)
            )
        instrument = Instrument(
            SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=20.0),
            (TransmissiveElement(Transmission([500, 600], [0.4, 0.8])),),
            coro,
            Detector(
                pixel_scale_arcsec=0.008,
                shape=(12, 12),
                qe_curve=([500, 600], [0.9, 0.7]),
            ),
        )
        scene = make_system(
            star=FlatStar(1e6, fixed_diameter_arcsec=0.013),
            smooth_planet=True,
            contrast=1e-7,
            separation_arcsec=0.02,
        )
        exposure = build_exposure(
            scene, instrument, (band,), time_d=0.0, order=2, field_lod=0.0
        )
        return scene, instrument, exposure

    return build


def gaussian_pixels(detector, frame, wavelength_nm, x_lod, y_lod, sigma_lod):
    """Independent unit-flux Gaussian from detector edges and the normal CDF."""
    pitch = float(detector.pixel_scale_arcsec) / float(
        frame.lod_to_arcsec(wavelength_nm)
    )
    ny, nx = detector.shape

    def probabilities(n, center):
        edges = (np.arange(n + 1) - n / 2) * pitch
        return np.diff(scipy_erf((edges - center) / (np.sqrt(2) * sigma_lod))) / 2

    return np.outer(probabilities(ny, y_lod), probabilities(nx, x_lod))
