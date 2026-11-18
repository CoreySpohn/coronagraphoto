"""Per-pixel readers of the shared photon and electron rate budgets.

Arrays are (band, node, y, x) for photons/s/nm and (band, y, x) for
band-integrated electrons/s. Geometry is concrete host metadata; readers cache
compiled kernels for each static configuration. Instruments and scenes can
be arguments of JAX transformations; an Exposure holds host geometry, so close
over it. A roll changes the position angle of detector +y, north through east.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache, partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image, Measure, PixelGrid, SkyOrientation, resample
from hwoutils.spectral import Band, Quadrature
from optixstuff import ElectronRates, Exposure, FramePlan, Instrument, PhotonRates
from optixstuff.coronagraph import AbstractImagingCoronagraph, AbstractSceneCoronagraph
from skyscapes import Scene


def _detector_grid(instrument, pointing):
    """Construct the one sky grid shared by planets, disks and observations."""
    if pointing.parity != -1:
        raise ValueError("detector pointing must have parity=-1")
    with jax.ensure_compile_time_eval():
        return PixelGrid.regular(
            instrument.detector.shape,
            instrument.detector.pixel_scale_arcsec,
            "arcsec",
            orientation=pointing,
        )


def _node_rates(
    wavelength_nm,
    system,
    instrument,
    time_d,
    grid,
    ephemeris,
    positions,
    zodi,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Render one node; detector yield never enters this photon calculation."""
    speckle = instrument.speckle
    coro = instrument.coronagraph.band(wavelength_nm)
    products = coro.at_wavelength(wavelength_nm)
    native = products.source_grid
    with jax.ensure_compile_time_eval():
        angular = replace(native.to_arcsec(wavelength_nm), orientation=grid.orientation)
    lod_arcsec = native.frame.lod_to_arcsec(wavelength_nm)
    gain = instrument.primary.collecting_area_m2 * instrument.system_throughput(
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
        sky = Image(
            jnp.broadcast_to(gain * radiance, grid.shape),
            grid,
            (),
            Measure.density("sr"),
            "ph s^-1 nm^-1",
        )
        local = sky.integrated().data * transmission

    disk = jnp.zeros(grid.shape)
    if system.disk is not None:
        radiance = system.photon_radiance(wavelength_nm, time_d, grid=grid)
        scene = radiance.integrated()
        scene = replace(resample(scene, angular), grid=native)
        rendered = coro.render_scene(scene, wavelength_nm, grid=detector_pixels(native))
        disk = gain * rendered.image.data
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
    instrument,
    time_d,
    pointing,
    quadrature,
    zodi,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Prepare static node geometry and solve the orbit once per observation."""
    grid = _detector_grid(instrument, pointing)
    nodes = np.asarray(quadrature.nodes_nm).ravel()
    ephemeris, positions = None, None
    if system.planets:
        times = jnp.atleast_1d(time_d)
        ephemeris = system.ephemeris(times)
        positions = system.positions(times, ephemeris=ephemeris)[:, :, 0]
    branches = tuple(
        partial(
            _node_rates,
            float(w),
            system,
            instrument,
            time_d,
            grid,
            ephemeris,
            positions,
            zodi,
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
    instrument,
    quadrature,
    time_d,
    pointing,
    zodi,
    ecliptic_latitude_deg,
    solar_longitude_deg,
):
    """Select concrete geometry and keep observation scalars dynamic."""
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(quadrature)):
        raise ValueError(
            "quadrature must be concrete host geometry; close over it under JIT"
        )
    coro = instrument.coronagraph
    for wavelength_nm in np.unique(np.asarray(quadrature.nodes_nm)):
        band = coro.band(float(wavelength_nm)) if coro is not None else None
        if not isinstance(band, AbstractImagingCoronagraph):
            raise TypeError("image formation requires an AbstractImagingCoronagraph")
        if system.disk is not None and not isinstance(band, AbstractSceneCoronagraph):
            raise TypeError("a system with a disk requires an AbstractSceneCoronagraph")
    return (
        system,
        instrument,
        jnp.asarray(time_d),
        pointing,
        zodi,
        jnp.asarray(ecliptic_latitude_deg),
        jnp.asarray(solar_longitude_deg),
    )


def _image_impl(
    system,
    instrument,
    time_d,
    pointing,
    zodi,
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
        instrument,
        time_d,
        pointing,
        quadrature,
        zodi,
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
            electrons = instrument.detector.electrons(photons)
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


def _run(scene, instrument, exposure, kind):
    """Run the shared kernel for one scene, instrument and exposure."""
    return _image_kernel(
        *_kernel_arguments(
            scene,
            instrument,
            exposure.quadrature,
            exposure.time_d,
            SkyOrientation(exposure.roll_deg),
            scene.zodi,
            exposure.ecliptic_latitude_deg,
            exposure.solar_longitude_deg,
        ),
        quadrature=_StaticQuadrature(exposure.quadrature),
        kind=kind,
    )


def build_exposure(
    scene: Scene,
    instrument: Instrument,
    bands: Sequence[Band],
    *,
    time_d,
    roll_deg=0.0,
    plan: FramePlan | None = None,
    order: int = 4,
) -> Exposure:
    """Build the host-side Exposure for one observation of a scene.

    Each band is split at the breaks the instrument and the scene's zodi
    declare, and an ``order``-point rule is applied to every piece (four nodes
    per band when no break falls inside it). The star is left out of the
    breaks. ``roll_deg`` is the position angle of detector +y, north through
    east, and must be a concrete number. ``plan`` is only needed by
    :func:`observe`. The local-zodi line of sight keeps the Exposure defaults
    (ecliptic latitude 0 deg, solar longitude 135 deg).
    """
    models = [m for m in (instrument, scene.zodi) if m is not None]
    return Exposure(
        time_d=time_d,
        roll_deg=roll_deg,
        quadrature=Quadrature.build(bands, *models, order=order),
        plan=plan,
    )


def photon_rates(scene: Scene, instrument: Instrument, exposure: Exposure):
    """Return full (W,Q,ny,nx) photon integrands, in photons/s/nm/pixel.

    One entry per source term and wavelength node of ``exposure.quadrature``.
    Call integrate() on the returned record to integrate its spectral nodes.
    No QE or survival is applied. Local zodi comes from ``scene.zodi`` (None
    disables it); the speckle field from ``instrument.speckle`` (None adds
    none). Thermal and companion terms are zero.
    """
    components = _run(scene, instrument, exposure, "photons")
    return PhotonRates(exposure.quadrature, **components)


def expected(scene: Scene, instrument: Instrument, exposure: Exposure) -> ElectronRates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    These are the expected rates for the instantaneous scene at
    ``exposure.time_d``; ``exposure.plan`` does not enter. Close over the
    host-built exposure under JIT. Each scan step sends one photon image and
    its original wavelength and weight to the detector. The eager cache reuses
    kernels for the same quadrature object. Static-grid branches grow compile
    size with Q. Forward storage is the band accumulator plus one node's
    images. JAX reverse-mode may retain residuals for differentiation.
    """
    return _run(scene, instrument, exposure, "electrons")


def observe(key, scene: Scene, instrument: Instrument, exposure: Exposure) -> Image:
    """Draw one electron Image (W,ny,nx), integrated on the detector sky grid.

    Rates describe the instantaneous scene at ``exposure.time_d``, held
    throughout ``exposure.plan``, which is required. Plan fields are scalar or
    (W,), with W matching the band axis. Detector noise is drawn once after
    combining all source terms.
    """
    if exposure.plan is None:
        raise ValueError("observe needs an exposure built with a FramePlan")
    electrons = expected(scene, instrument, exposure)
    return Image(
        _image_kernel(
            instrument.detector, key, electrons, exposure.plan, _function=_draw_impl
        ),
        _detector_grid(instrument, SkyOrientation(exposure.roll_deg)),
        ("band",),
        Measure.integral(),
        "electrons",
    )
