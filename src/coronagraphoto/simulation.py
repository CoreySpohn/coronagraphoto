"""Per-pixel readers of the shared photon and electron rate budgets.

Arrays are (band, node, y, x) for photons/s/nm and (band, y, x) for
band-integrated electrons/s. Geometry is concrete host metadata; readers cache
compiled kernels for each static configuration. Optical paths and scenes can
be arguments of JAX transformations. A roll changes the
position angle of detector +y, north through east, in SkyOrientation.
"""

from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache, partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import arcsec2rad
from hwoutils.grids import Image, PixelGrid, SkyOrientation, resample
from hwoutils.spectral import Band, Quadrature
from optixstuff import ElectronRates, FramePlan, OpticalPath, PhotonRates
from optixstuff.coronagraph import AbstractTableCoronagraph, MultiBandCoronagraph
from optixstuff.speckle import AbstractSpeckleField
from skyscapes.scene import System
from zodi.jax import AbstractLocalZodi


def _detector_grid(optical_path, pointing):
    """Construct the one sky grid shared by planets, disks and observations."""
    if pointing.parity != -1:
        raise ValueError("detector pointing must have parity=-1")
    with jax.ensure_compile_time_eval():
        return PixelGrid.regular(
            optical_path.detector.shape,
            optical_path.detector.pixel_scale_arcsec,
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
        stellar_grid = coro.stellar_map(wavelength_nm, 0.0, grid=None).grid
        background_grid = coro.sky_trans.grid
    with jax.ensure_compile_time_eval():
        angular = replace(native.to_arcsec(wavelength_nm), orientation=grid.orientation)
    lod_arcsec = native.frame.lod_to_arcsec(wavelength_nm)
    gain = optical_path.primary.collecting_area_m2 * optical_path.system_throughput(
        wavelength_nm
    )

    def detector_pixels(product_grid):
        # The detector pixels in a product's lambda/D frame at this node, so
        # the product integrates once over each detector pixel. Native axes
        # are detector axes; only the scale changes with wavelength.
        with jax.ensure_compile_time_eval():
            scale = float(product_grid.frame.lod_to_arcsec(wavelength_nm))
            return replace(
                grid,
                basis=tuple(tuple(b / scale for b in row) for row in grid.basis),
                unit="lod",
                frame=product_grid.frame,
                orientation=product_grid.orientation,
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
    diameter = star.angular_diameter_arcsec(system.dist_pc)
    leakage = coro.stellar_map(
        wavelength_nm,
        diameter / stellar_grid.frame.lod_to_arcsec(wavelength_nm),
        grid=detector_pixels(stellar_grid),
    ).data
    if speckle is not None:
        # The field owns its native lod grid, Frame, reconstruction and
        # extension; its clock subtracts its own epoch. The excess is signed
        # (coherent interference can lower the floor); only the total
        # leakage is clipped at zero below.
        leakage = leakage + onto_detector(
            speckle.realize(wavelength_nm=wavelength_nm, time_d=time_d)
        )
    leakage = gain * phi_star * jnp.maximum(leakage, 0.0)

    planet = jnp.zeros(grid.shape)
    if system.planets:
        x, y = grid.orientation.sky_to_detector(positions[0], positions[1])
        phi = system.planet_flux_densities(
            wavelength_nm, jnp.atleast_1d(time_d), ephemeris=ephemeris
        )[:, 0]
        psfs = coro.source_psfs(
            wavelength_nm, x / lod_arcsec, y / lod_arcsec, grid=detector_pixels(native)
        )
        planet = gain * jnp.einsum("k,kyx->yx", phi, psfs.data)

    omega_sr = grid.pixel_area * arcsec2rad**2
    local = jnp.zeros(grid.shape)
    if zodi is not None:
        radiance = zodi.photon_radiance(
            wavelength_nm,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
        transmission = coro.background_transmission(
            wavelength_nm, grid=detector_pixels(background_grid)
        ).data
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
        else:
            quadrature = Quadrature(
                *(
                    jnp.asarray(np.frombuffer(data, dtype=dtype).reshape(shape))
                    for shape, dtype, data in quadrature
                )
            )
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


def _kernel_arguments(
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
    """Freeze host quadrature by value and keep observation scalars dynamic."""
    if quadrature is not None:
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(quadrature)):
            raise ValueError(
                "quadrature must be concrete host geometry; close over it under JIT"
            )
        quadrature = tuple(
            (a.shape, a.dtype.str, a.tobytes())
            for a in map(np.asarray, (quadrature.nodes_nm, quadrature.weights_nm))
        )
    return (
        system,
        optical_path,
        tuple(bands),
        jnp.asarray(time_d),
        pointing,
        quadrature_order,
        quadrature,
        zodi,
        speckle,
        jnp.asarray(ecliptic_latitude_deg),
        jnp.asarray(solar_longitude_deg),
    )


def _image_impl(
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
    *,
    kind,
):
    """Render one configuration with array leaves supplied at runtime."""
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
    if kind == "photons":
        values = jax.lax.map(
            lambda i: jax.lax.switch(i, branches), jnp.arange(len(branches))
        )
        return jax.tree.map(
            lambda x: x.reshape((*quadrature.nodes_nm.shape, *grid.shape)), values
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

    electrons = jax.lax.scan(accumulate, initial, jnp.arange(len(branches)))[0]
    return electrons


def _draw_impl(detector, key, electrons, plan):
    """Stage detector noise separately so observations share the renderer."""
    return detector.draw(key, electrons, plan)


@lru_cache(maxsize=4)
def _cached_image_kernel(function, treedef, static_leaves, signature):
    """Own at most four eager JIT functions, including shape/dtype variants."""
    static = jax.tree.unflatten(treedef, static_leaves)

    def run(dynamic):
        args, kwargs = eqx.combine(dynamic, static)
        return function(*args, **kwargs)

    # Only arrays are passed to jit; host metadata is in the bounded closure.
    # A separate function per entry lets eviction release its JAX cache too.
    return jax.jit(run)


def _image_kernel(*args, _function=_image_impl, **kwargs):
    """Reuse eager kernels; let caller transformations own their staging cache."""
    dynamic, static = eqx.partition((args, kwargs), eqx.is_array)
    arrays = jax.tree.leaves(dynamic)
    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        return _function(*args, **kwargs)
    static_leaves, treedef = jax.tree.flatten(static)
    signature = tuple(
        (a.shape, a.dtype, getattr(a, "weak_type", False)) for a in arrays
    )
    return _cached_image_kernel(_function, treedef, tuple(static_leaves), signature)(
        dynamic
    )


def clear_reader_cache():
    """Release all eager reader kernels; the next eager call compiles again.

    Caller-owned JIT executables and live output arrays are unaffected. JAX
    and the system allocator can retain memory after executable release.
    """
    _cached_image_kernel.cache_clear()


def image_rates(
    system: System,
    optical_path: OpticalPath,
    bands: Sequence[Band],
    *,
    time_d,
    pointing: SkyOrientation,
    quadrature_order: int = 4,
    quadrature: Quadrature | None = None,
    zodi: AbstractLocalZodi | None = None,
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
    return _image_kernel(
        *_kernel_arguments(
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
        ),
        kind="photons",
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
    zodi: AbstractLocalZodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> ElectronRates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    quadrature, when supplied, owns the nodes and weights and supersedes
    quadrature_order. Its arrays become a static key by value; close over
    this host-prepared geometry under JIT. OpticalPath may be an argument.
    Each scan step wraps one photon image in (1,1,ny,nx), passes its one-node
    Quadrature to detector.electrons, and adds that result to its band. No
    node images are stacked. Static-grid branches grow compile size with Q;
    forward image storage is the band accumulator plus one node's images.
    JAX reverse-mode may retain residuals for differentiation.
    """
    return _image_kernel(
        *_kernel_arguments(
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
        ),
        kind="electrons",
    )


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
    zodi: AbstractLocalZodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> Image:
    """Draw one electron Image (W,ny,nx), per_pixel on the detector sky grid.

    Rates describe the instantaneous scene at time_d, held throughout plan.
    Plan fields are scalar or (W,), with W matching the band axis.
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
        _image_kernel(
            optical_path.detector, key, electrons, plan, _function=_draw_impl
        ),
        _detector_grid(optical_path, pointing),
        ("band",),
        "per_pixel",
        "electrons",
    )
