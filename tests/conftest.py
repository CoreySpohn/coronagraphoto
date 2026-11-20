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
from optixstuff import ConstantThroughput, Detector, Instrument, SimplePrimary
from optixstuff.apertures import Circular
from optixstuff.coronagraph import (
    AbstractSceneCoronagraph,
    AbstractTableCoronagraph,
    SceneRender,
    SourcePSFs,
)
from optixstuff.grids import Frame, Image, Measure, PixelGrid, resample
from optixstuff.performance import PerformanceCurves, SampledProfile
from optixstuff.spectral import Band
from orbix.observatory import ecliptic_pointing
from skyscapes.physical_model import GridPhysicalModel
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
    chromatic: bool = eqx.field(static=True, default=False)

    def __init__(self, n=40, step=0.25, *, chromatic=False, frame=None, bandpass=None):
        """Build an n x n focal grid with the given lambda/D step."""
        if frame is None:
            frame = Frame(550.0, 6.0, "circumscribed")
        self.bandpass = bandpass
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

    def _native_stellar_map(self, stellar_diam_lod):
        """Uniform leakage density with a resolved-diameter response."""
        return Image(
            jnp.full(self.grid.shape, 1e-9 * self.grid.pixel_area)
            * (1 + stellar_diam_lod**2),
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
        return PerformanceCurves(
            [0.0],
            self.grid.frame,
            rule,
            [tau],
            [area],
            [1e-9 * area],
            [1e-9],
            [1e-9 * area / tau],
            0.0,
            min(self.grid.shape) * self.grid.step / 2,
            throughput_variance=[tau],
            core_area_variance_lod2=[area],
            leakage_fraction_variance=[1e-9 * area],
            leakage_density_variance_lod2=[1e-9],
        )


class GaussianSceneCoronagraph(GaussianCoronagraph, AbstractSceneCoronagraph):
    """Full dense operator built from the same synthetic off-axis PSFs."""

    operator: object

    def __init__(self, n=40, step=0.25, *, chromatic=False):
        """Prepare a full set of native Gaussian source responses."""
        from yippy import DenseSceneOperator

        super().__init__(n, step, chromatic=chromatic)
        x, y = self.grid.coordinates()
        psfs = self.source_psfs(550.0, x.ravel(), y.ravel(), grid=None)
        self.operator = DenseSceneOperator(
            psfs.image.data.reshape(n, n, n, n), psfs.valid.reshape(n, n)
        )

    def render_scene(self, scene, wavelength_nm, *, grid):
        """Apply the production dense operator to per-pixel incident flux."""
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
):
    """Real static-spectrum photometry with a circular orbit at its node."""
    star = FlatStar(flux_phot_per_nm_m2=1e6) if star is None else star
    wavelengths = jnp.array([400.0, 500.0, 600.0, 700.0])
    contrast = jnp.asarray(contrast)
    values = jnp.broadcast_to(contrast, (4,))
    model = GridPhysicalModel(
        wavelengths,
        jnp.array([0.0, 90.0, 180.0]),
        jnp.broadcast_to(values[None, :, None], (1, 4, 3)),
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
        a_AU=jnp.atleast_1d(separation_arcsec) * 10,
        e=0.0,
        W_rad=jnp.deg2rad(sky_pa_deg),
        i_rad=0.0,
        w_rad=0.0,
        M0_rad=0.0,
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
        qe_curve=(jnp.array([400.0, 700.0]), jnp.array([1.1, 0.5]).clip(0, 1)),
        survival_fraction=0.9,
    )
    # Exact .8 - .002*(lambda-550) in the worked interval.
    detector = eqx.tree_at(
        lambda d: d._qe_curve,
        detector,
        (jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
    )
    return Instrument(
        SimplePrimary(circumscribed_diameter_m=6.0, collecting_area_m2=40.0),
        (ConstantThroughput(0.4),),
        coro,
        detector,
    )


def worked_setup(*, n_wavelengths=1201):
    """AB=5 flat F_nu, dense cubic photon interpolation, linear contrast."""
    wave = jnp.linspace(400.0, 700.0, n_wavelengths)
    star = Star(
        wavelengths_nm=wave,
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.full((wave.size, 1), 3631.0 * 10 ** (-0.4 * 5)),
    )
    contrast = 1e-10 * (1 - 0.004 * (jnp.array([400.0, 500.0, 600.0, 700.0]) - 550))
    return make_system(star=star, contrast=contrast), make_path(
        chromatic=True, detector_n=48, detector_step=0.3
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
        return PerformanceCurves(
            [0.0],
            self.grid.frame,
            rule,
            [tau],
            [area],
            [ell],
            [1e-9],
            [ell / tau],
            0.0,
            min(self.grid.shape) * self.grid.step / 2,
            throughput_variance=[tau],
            core_area_variance_lod2=[area],
            leakage_fraction_variance=[ell],
            leakage_density_variance_lod2=[1e-9],
        )


def write_scene(path):
    """Small ExoVista v2.6 FITS with a native disk and independent sky states."""
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
    planet[:, 9] = 1.0
    planet[:, 13] = np.sqrt(G_si * (Msun2kg + Mearth2kg) / AU2m) * yr2s / AU2m
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
                planet, header=fits.Header({"R": 1.0, "M": 1.0, "A": 1.0, "E": 0.0})
            ),
        ]
    ).writeto(path)
    return str(path)


def write_yip(path, *, distinct_products=False):
    """Synthetic YIP package with SI-labelled wavelength headers and even centre."""
    from yippy.testing import write_synthetic_yip
    from yippy.writer import yip_header

    n, step, sigma = 24, 0.5, 0.7
    yy, xx = np.indices((n, n))
    xx, yy = (xx - (n - 1) / 2) * step, (yy - (n - 1) / 2) * step
    offsets = np.array(
        [(x, y) for x in (0.0, 1.0, 2.0, 3.0) for y in (0.0, 1.0, 2.0, 3.0)]
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
    sky = np.full((n, n), 0.3)
    metadata = dict(
        inscribed_diameter_m=4.7,
        obscured=0.2,
        low_nm=500.0,
        high_nm=600.0,
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
