"""Per-pixel readers of the shared photon and electron rate budgets.

Arrays are (band, node, y, x) for photons/s/nm and (band, y, x) for
band-integrated electrons/s. Geometry is concrete host metadata. Close over
instrument geometry, bands and pointing when applying JAX transformations;
scene parameters and observation time can be traced. A roll changes the
position angle of detector +y, north through east, in SkyOrientation.
"""

from collections.abc import Sequence
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import arcsec2rad
from hwoutils.grids import Image, PixelGrid, SkyOrientation, resample
from hwoutils.spectral import Band, Quadrature
from optixstuff import ElectronRates, FramePlan, OpticalPath, PhotonRates
from optixstuff.coronagraph import AbstractTableCoronagraph, MultiBandCoronagraph
from optixstuff.speckle import AbstractSpeckleField
from skyscapes.background import Zodi
from skyscapes.scene import System


def _detector_grid(optical_path, pointing):
    """Construct the one sky grid shared by planets, disks and observations."""
    if pointing.parity != -1:
        raise ValueError("detector pointing must have parity=-1")
    with jax.ensure_compile_time_eval():
        return PixelGrid.regular(
            optical_path.detector.shape,
            float(optical_path.detector.pixel_scale_arcsec),
            "arcsec",
            orientation=pointing,
        )


def _node_rates(
    wavelength_nm,
    system,
    optical_path,
    time_d,
    grid,
    ephemeris,
    positions,
    zodi,
    speckle,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Render one node; detector yield never enters this photon calculation."""
    coro = optical_path.coronagraph
    if isinstance(coro, MultiBandCoronagraph):
        coro = coro.band(wavelength_nm)
    native = coro.grid
    stellar_grid = native
    background_grid = native
    if isinstance(coro, AbstractTableCoronagraph):
        stellar_grid = coro.stellar_intens(0.0).grid
        background_grid = coro.sky_trans.grid
    with jax.ensure_compile_time_eval():
        angular = replace(native.to_arcsec(wavelength_nm), orientation=grid.orientation)
    lod_arcsec = native.frame.lod_to_arcsec(wavelength_nm)
    gain = optical_path.primary.collecting_area_m2 * optical_path.system_throughput(
        wavelength_nm
    )

    def onto_detector(image):
        # Native coordinates describe detector axes, so attach the live pointing
        # only after the coronagraph has rendered its instrumental response.
        with jax.ensure_compile_time_eval():
            product_grid = replace(
                image.grid.to_arcsec(wavelength_nm), orientation=grid.orientation
            )
        return resample(replace(image, grid=product_grid), grid).data

    star = system.star
    phi_star = star.photon_flux_density(wavelength_nm, time_d, dist_pc=system.dist_pc)
    diameter = star.diameter_arcsec
    if callable(diameter):
        diameter = diameter(dist_pc=system.dist_pc)
    leakage = onto_detector(
        coro.stellar_map(
            wavelength_nm,
            diameter / stellar_grid.frame.lod_to_arcsec(wavelength_nm),
            grid=stellar_grid,
        )
    )
    if speckle is not None:
        delta = speckle.realize(wavelength_nm=wavelength_nm, time_d=time_d)
        with jax.ensure_compile_time_eval():
            speckle_grid = PixelGrid.regular(
                delta.shape,
                speckle.pixel_scale_lod,
                "lod",
                frame=optical_path.primary.frame(native.frame.reference_wavelength_nm),
            ).to_arcsec(wavelength_nm)
            speckle_grid = replace(speckle_grid, orientation=grid.orientation)
        delta = resample(
            Image(
                delta,
                speckle_grid,
                (),
                "per_pixel",
                "stellar_flux_fraction",
                outside="reflect",
            ),
            grid,
        ).data
        # Preserve the existing positive excess-halo treatment, without changing
        # the speckle evolution model or subtracting its epoch a second time.
        leakage = leakage + jnp.maximum(delta, 0.0)
    leakage = gain * phi_star * jnp.maximum(leakage, 0.0)

    planet = jnp.zeros(grid.shape)
    if system.planets:
        x, y = grid.orientation.sky_to_detector(positions[0], positions[1])
        phi = system.planet_flux_densities(
            wavelength_nm, jnp.atleast_1d(time_d), ephemeris=ephemeris
        )[:, 0]
        psfs = coro.source_psfs(
            wavelength_nm, x / lod_arcsec, y / lod_arcsec, grid=native
        )
        planet = gain * jnp.einsum("k,kyx->yx", phi, onto_detector(psfs))

    omega_sr = grid.pixel_area * arcsec2rad**2
    local = jnp.zeros(grid.shape)
    if zodi is not None:
        radiance = zodi.photon_radiance(
            wavelength_nm,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
        transmission = onto_detector(
            coro.background_transmission(wavelength_nm, grid=background_grid)
        )
        local = gain * radiance * omega_sr * transmission

    disk = jnp.zeros(grid.shape)
    if system.disk is not None:
        radiance = system.photon_radiance(wavelength_nm, time_d, grid=grid)
        scene = Image(
            radiance.data * omega_sr,
            grid,
            (),
            "per_pixel",
            "ph s^-1 m^-2 nm^-1",
            reconstruction="linear",
        )
        scene = replace(resample(scene, angular), grid=native)
        # Convolution with the band-limited PSF band-limits the output. A finite
        # scene is zero outside; a nonzero extension follows the input scene.
        disk = gain * onto_detector(
            replace(
                coro.extended_scene(scene, wavelength_nm, grid=native),
                reconstruction="band_limited",
                outside=scene.outside,
            )
        )
    zero = jnp.zeros(grid.shape)
    return PhotonRates(planet, leakage, local, disk, zero, zero)


def _reader(
    system,
    optical_path,
    bands,
    time_d,
    pointing,
    quadrature_order,
    quadrature,
    zodi,
    speckle,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Prepare static node geometry and solve the orbit once per observation."""
    grid = _detector_grid(optical_path, pointing)
    with jax.ensure_compile_time_eval():
        if quadrature is None:
            quadrature = Quadrature.build(bands, None, quadrature_order)
        if quadrature.nodes_nm.shape[0] != len(bands):
            raise ValueError("quadrature must have one row per band")
        nodes = np.asarray(quadrature.nodes_nm).ravel()
    ephemeris, positions = None, None
    if system.planets:
        times = jnp.atleast_1d(time_d)
        ephemeris = system.ephemeris(times)
        positions = system.positions(times, ephemeris=ephemeris)[:, :, 0]
    speckle = optical_path.speckle if speckle is None else speckle
    branches = tuple(
        partial(
            _node_rates,
            float(w),
            system,
            optical_path,
            time_d,
            grid,
            ephemeris,
            positions,
            zodi,
            speckle,
            ecliptic_latitude_deg,
            solar_longitude_deg,
        )
        for w in nodes
    )
    return grid, quadrature, branches


def image_rates(
    system: System,
    optical_path: OpticalPath,
    bands: Sequence[Band],
    *,
    time_d,
    pointing: SkyOrientation,
    quadrature_order: int = 4,
    quadrature: Quadrature | None = None,
    zodi: Zodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> PhotonRates:
    """Return full (W,Q,ny,nx) photon integrands, in photons/s/nm/pixel.

    Pass a host-prepared quadrature to share nodes and weights with other
    readers; otherwise use Quadrature.build(bands, None, quadrature_order)
    to integrate this record. The default is unsplit Gauss-Legendre: subdivide
    bands at spectral breaks and increase order to resolve spectral structure.
    No QE or survival is applied. zodi=None disables local zodi; speckle=None
    uses the path's optional speckle field. Thermal and companion terms are zero.
    """
    grid, quadrature, branches = _reader(
        system,
        optical_path,
        bands,
        time_d,
        pointing,
        quadrature_order,
        quadrature,
        zodi,
        speckle,
        ecliptic_latitude_deg,
        solar_longitude_deg,
    )
    values = jax.lax.map(
        lambda i: jax.lax.switch(i, branches), jnp.arange(len(branches))
    )
    return jax.tree.map(
        lambda x: x.reshape((*quadrature.nodes_nm.shape, *grid.shape)), values
    )


def image_electrons(
    system: System,
    optical_path: OpticalPath,
    bands: Sequence[Band],
    *,
    time_d,
    pointing: SkyOrientation,
    quadrature_order: int = 4,
    quadrature: Quadrature | None = None,
    zodi: Zodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> ElectronRates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    quadrature, when supplied, owns the nodes and weights and supersedes
    quadrature_order. Close over this host-prepared geometry under JIT.
    Each scan step wraps one photon image in (1,1,ny,nx), passes its one-node
    Quadrature to detector.electrons, and adds that result to its band. No
    node images are stacked. Static-grid branches grow compile size with Q;
    forward image storage is the band accumulator plus one node's images.
    JAX reverse-mode may retain residuals for differentiation.
    """
    grid, quadrature, branches = _reader(
        system,
        optical_path,
        bands,
        time_d,
        pointing,
        quadrature_order,
        quadrature,
        zodi,
        speckle,
        ecliptic_latitude_deg,
        solar_longitude_deg,
    )
    n_bands, n_nodes = quadrature.nodes_nm.shape
    initial = ElectronRates(
        **{
            name: jnp.zeros((n_bands, *grid.shape))
            for name in PhotonRates.__dataclass_fields__
        }
    )

    def accumulate(total, index):
        photons = jax.lax.switch(index, branches)
        photons = jax.tree.map(lambda x: x[None, None], photons)
        single = Quadrature(
            quadrature.nodes_nm.reshape(-1)[index].reshape(1, 1),
            quadrature.weights_nm.reshape(-1)[index].reshape(1, 1),
        )
        electrons = optical_path.detector.electrons(photons, single)
        total = jax.tree.map(
            lambda a, b: a.at[index // n_nodes].add(b[0]), total, electrons
        )
        return total, None

    return jax.lax.scan(accumulate, initial, jnp.arange(len(branches)))[0]


def observe(
    key,
    system: System,
    optical_path: OpticalPath,
    bands: Sequence[Band],
    plan: FramePlan,
    *,
    time_d,
    pointing: SkyOrientation,
    quadrature_order: int = 4,
    quadrature: Quadrature | None = None,
    zodi: Zodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> Image:
    """Draw one electron Image (W,ny,nx), per_pixel on the detector sky grid.

    Rates describe the instantaneous scene at time_d, held throughout plan.
    Detector noise is drawn once after combining all source terms.
    """
    electrons = image_electrons(
        system,
        optical_path,
        bands,
        time_d=time_d,
        pointing=pointing,
        quadrature_order=quadrature_order,
        quadrature=quadrature,
        zodi=zodi,
        speckle=speckle,
        ecliptic_latitude_deg=ecliptic_latitude_deg,
        solar_longitude_deg=solar_longitude_deg,
    )
    return Image(
        optical_path.detector.draw(key, electrons, plan),
        _detector_grid(optical_path, pointing),
        ("band",),
        "per_pixel",
        "electrons",
    )
