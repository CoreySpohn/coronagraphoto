"""Per-pixel readers of the shared photon and electron rate budgets.

Arrays are (band, node, y, x) for photons/s/nm and (band, y, x) for
band-integrated electrons/s. Geometry is concrete host metadata; readers cache
compiled kernels for each static configuration. Optical paths and scenes can
be arguments of JAX transformations. A roll changes the
position angle of detector +y, north through east, in SkyOrientation.
"""

from dataclasses import dataclass, replace
from functools import lru_cache, partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import arcsec2rad
from hwoutils.grids import Image, PixelGrid, SkyOrientation, resample
from hwoutils.spectral import Quadrature
from optixstuff import ElectronRates, FramePlan, OpticalPath, PhotonRates
from optixstuff.coronagraph import AbstractImagingCoronagraph, AbstractSceneCoronagraph
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
    coro = optical_path.coronagraph.band(wavelength_nm)
    products = coro.at_wavelength(wavelength_nm)
    native = products.source_grid
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
            return replace(
                grid.to_lod(wavelength_nm, product_grid.frame),
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
        diameter / products.stellar_grid.frame.lod_to_arcsec(wavelength_nm),
        grid=detector_pixels(products.stellar_grid),
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
        # A dark source contributes zero even if its PSF is undefined.
        samples = jnp.where(psfs.valid[:, None, None], psfs.image.data, 0.0)
        planet = gain * jnp.einsum("k,kyx->yx", phi, samples)
        planet = jnp.where(jnp.all(psfs.valid | (phi == 0)), planet, jnp.nan)

    omega_sr = grid.pixel_area * arcsec2rad**2
    local = jnp.zeros(grid.shape)
    if zodi is not None:
        radiance = zodi.photon_radiance(
            wavelength_nm,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
        transmission = coro.uniform_sky_map(
            wavelength_nm, grid=detector_pixels(products.sky_grid)
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
        rendered = coro.render_scene(scene, wavelength_nm, grid=native)
        disk = gain * onto_detector(
            replace(
                rendered.image,
                reconstruction="band_limited",
                outside=scene.outside,
            )
        )
        disk = jnp.where(rendered.valid, disk, jnp.nan)
    zero = jnp.zeros(grid.shape)
    return dict(
        planet=planet,
        star_leakage=leakage,
        local_zodi=local,
        exozodi=disk,
        thermal=zero,
        stellar_companions=zero,
    )


def _reader(
    system,
    optical_path,
    time_d,
    pointing,
    quadrature,
    zodi,
    speckle,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Prepare static node geometry and solve the orbit once per observation."""
    grid = _detector_grid(optical_path, pointing)
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
    return grid, branches


@dataclass(frozen=True, eq=False)
class _StaticQuadrature:
    """Keep the concrete wavelength geometry in the kernel closure."""

    rule: Quadrature

    def __hash__(self):
        return id(self.rule)

    def __eq__(self, other):
        return isinstance(other, _StaticQuadrature) and self.rule is other.rule


def _kernel_arguments(
    system,
    optical_path,
    quadrature,
    time_d,
    pointing,
    zodi,
    speckle,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Select concrete geometry and keep observation scalars dynamic."""
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(quadrature)):
        raise ValueError(
            "quadrature must be concrete host geometry; close over it under JIT"
        )
    coro = optical_path.coronagraph
    for wavelength_nm in np.unique(np.asarray(quadrature.nodes_nm)):
        band = coro.band(float(wavelength_nm)) if coro is not None else None
        if not isinstance(band, AbstractImagingCoronagraph):
            raise TypeError("image formation requires an AbstractImagingCoronagraph")
        if system.disk is not None and not isinstance(band, AbstractSceneCoronagraph):
            raise TypeError("a system with a disk requires an AbstractSceneCoronagraph")
    return (
        system,
        optical_path,
        jnp.asarray(time_d),
        pointing,
        zodi,
        speckle,
        jnp.asarray(ecliptic_latitude_deg),
        jnp.asarray(solar_longitude_deg),
    )


def _image_impl(
    system,
    optical_path,
    time_d,
    pointing,
    zodi,
    speckle,
    ecliptic_latitude_deg,
    solar_longitude_deg,
    *,
    quadrature,
    kind,
    host_loop=False,
):
    """Render the same concrete node geometries under JIT or host iteration."""
    quadrature = quadrature.rule
    grid, branches = _reader(
        system,
        optical_path,
        time_d,
        pointing,
        quadrature,
        zodi,
        speckle,
        ecliptic_latitude_deg,
        solar_longitude_deg,
    )
    if kind == "photons":
        if host_loop:
            values = jax.tree.map(
                lambda *x: jnp.stack(x), *(read() for read in branches)
            )
        else:
            values = jax.lax.map(
                lambda i: jax.lax.switch(i, branches), jnp.arange(len(branches))
            )
        return jax.tree.map(
            lambda x: x.reshape((*quadrature.nodes_nm.shape, *grid.shape)), values
        )

    # Grid metadata is static, so stage the concrete wavelength branches of
    # each band. Only its accumulated rate and one node image are live.
    n_nodes = quadrature.nodes_nm.shape[1]

    def render_band(i):
        row = quadrature.band(i)
        band_branches = branches[i * n_nodes : (i + 1) * n_nodes]
        initial = {
            name: jnp.zeros(grid.shape) for name in ElectronRates.component_names()
        }

        def accumulate(total, j, components):
            single = Quadrature(
                row.bands, row.nodes_nm[:, j, None], row.weights_nm[:, j, None]
            )
            photons = PhotonRates(
                single, **jax.tree.map(lambda x: x[None, None], components)
            )
            electrons = optical_path.detector.electrons(photons)
            return {name: a + getattr(electrons, name)[0] for name, a in total.items()}

        if host_loop:
            total = initial
            for j, read in enumerate(band_branches):
                total = accumulate(total, j, read())
            return total

        def step(total, j):
            return accumulate(total, j, jax.lax.switch(j, band_branches)), None

        return jax.lax.scan(step, initial, jnp.arange(n_nodes))[0]

    rows = [render_band(i) for i in range(len(quadrature.bands))]
    return ElectronRates(
        quadrature.bands, **jax.tree.map(lambda *x: jnp.stack(x), *rows)
    )


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
    # Each static wavelength needs its own resampling geometry and switch
    # branch. A dense spectrum can have thousands of nodes. Keep the eager
    # compiler footprint bounded by dispatching those node operations directly.
    quadrature = kwargs.get("quadrature")
    if quadrature is not None and quadrature.rule.nodes_nm.size > 32:
        return _function(*args, **kwargs, host_loop=True)
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
    quadrature: Quadrature,
    *,
    time_d,
    pointing: SkyOrientation,
    zodi: AbstractLocalZodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> PhotonRates:
    """Return full (W,Q,ny,nx) photon integrands, in photons/s/nm/pixel.

    Build quadrature on the host from the system, path and active backgrounds.
    Call integrate() on the returned record to integrate its spectral nodes.
    No QE or survival is applied. zodi=None disables local zodi; speckle=None
    uses the path's optional speckle field. Thermal and companion terms are zero.
    """
    components = _image_kernel(
        *_kernel_arguments(
            system,
            optical_path,
            quadrature,
            time_d,
            pointing,
            zodi,
            speckle,
            ecliptic_latitude_deg,
            solar_longitude_deg,
        ),
        quadrature=_StaticQuadrature(quadrature),
        kind="photons",
    )
    return PhotonRates(quadrature, **components)


def image_electrons(
    system: System,
    optical_path: OpticalPath,
    quadrature: Quadrature,
    *,
    time_d,
    pointing: SkyOrientation,
    zodi: AbstractLocalZodi | None = None,
    speckle: AbstractSpeckleField | None = None,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
) -> ElectronRates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    Close over the host-prepared quadrature under JIT. Each scan step sends
    one photon image and its original wavelength and weight to the detector.
    The eager cache reuses kernels for the same quadrature object. Static-grid
    branches grow compile size with Q. Forward storage is the band accumulator
    plus one node's images.
    JAX reverse-mode may retain residuals for differentiation.
    """
    return _image_kernel(
        *_kernel_arguments(
            system,
            optical_path,
            quadrature,
            time_d,
            pointing,
            zodi,
            speckle,
            ecliptic_latitude_deg,
            solar_longitude_deg,
        ),
        quadrature=_StaticQuadrature(quadrature),
        kind="electrons",
    )


def observe(
    key,
    system: System,
    optical_path: OpticalPath,
    quadrature: Quadrature,
    plan: FramePlan,
    *,
    time_d,
    pointing: SkyOrientation,
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
        quadrature,
        time_d=time_d,
        pointing=pointing,
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
