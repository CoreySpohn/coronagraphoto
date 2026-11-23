"""Per-pixel readers of the shared photon and electron rate budgets.

Arrays are (band, node, y, x) for photons/s/nm and (band, y, x) for
band-integrated electrons/s. Detector shape, grid axes, frame and band
selection are static; the wavelength nodes and weights, the roll and the epoch
are traced leaves, so one compiled program serves any node values and any roll
(a new node count or band count is a new array shape and compiles once).
Scenes, instruments and exposures can all be arguments of JAX
transformations. Every node is rendered through resampling operators built
from the traced lambda/D scale. A roll changes the position angle of detector
+y, north through east.
"""

from collections.abc import Sequence
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from optixstuff import Exposure, FramePlan, Instrument, Rates
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    AbstractSceneCoronagraph,
    MultiBandCoronagraph,
)
from optixstuff.grids import (
    Image,
    Measure,
    PixelGrid,
    SkyOrientation,
    coverage,
    footprint_overlap,
    integrate_footprints,
    reconstruction_margin,
    resample,
)
from optixstuff.spectral import Band, Quadrature
from skyscapes import Scene


def _detector_grid(instrument, orientation):
    """Construct the one sky grid shared by planets, disks and observations."""
    if orientation is not None and orientation.parity != -1:
        raise ValueError("detector pointing must have parity=-1")
    return replace(instrument.detector.grid, orientation=orientation)


# Disk sub-samples per source-cell side; see ``_refined``.
_DISK_SUBSAMPLE = 4


def _refined(grid, factor):
    """The same field of view cut into ``factor`` times finer cells per side."""
    (bxx, bxy), (byx, byy) = grid.basis
    return PixelGrid(
        tuple(n * factor for n in grid.shape),
        (grid.center_x + 0.5) * factor - 0.5,
        (grid.center_y + 0.5) * factor - 0.5,
        ((bxx / factor, bxy / factor), (byx / factor, byy / factor)),
        grid.unit,
        grid.frame,
        grid.orientation,
    )


def _node_rates(
    wavelength_nm,
    coro,
    system,
    instrument,
    time_d,
    grid,
    ephemeris,
    positions,
    sky_radiance,
):
    """Render one node; detector yield never enters this photon calculation.

    ``sky_radiance`` is the local-zodi radiance at this node in photons s^-1
    m^-2 nm^-1 sr^-1, or None for no local zodi. Returns the component images
    and the boolean (ny, nx) of detector pixels the coronagraph data cover at
    this node: the AND of ``coverage`` over every product the render reads,
    each with the reach of its own reconstruction. Nothing is reflected or
    extrapolated, so the light at an uncovered pixel is not modeled; the
    caller zeroes it.
    """
    speckle = instrument.speckle
    products = coro.at_wavelength(wavelength_nm)
    native = products.source_grid
    lod_arcsec = native.frame.lod_to_arcsec(wavelength_nm)
    gain = instrument.effective_area_m2(wavelength_nm)

    def detector_pixels(product_grid):
        # The detector pixels in a product's lambda/D frame at this node, so
        # the product integrates once over each detector pixel. Native axes
        # are detector axes; only the scale changes with wavelength.
        return replace(
            grid.to_lod(wavelength_nm, product_grid.frame),
            orientation=product_grid.orientation,
        )

    def onto_detector(image):
        # Native coordinates describe detector axes, so attach the live pointing
        # only after the coronagraph has rendered its instrumental response.
        product_grid = replace(
            image.grid.to_arcsec(wavelength_nm), orientation=grid.orientation
        )
        return resample(replace(image, grid=product_grid), grid).data

    star = system.star
    phi_star = star.photon_flux_density(wavelength_nm, time_d, dist_pc=system.dist_pc)
    diameter = star.angular_diameter_arcsec(system.dist_pc)
    stellar = coro.stellar_map(
        wavelength_nm,
        diameter / products.stellar_grid.frame.lod_to_arcsec(wavelength_nm),
        grid=detector_pixels(products.stellar_grid),
    )
    leakage = stellar.data
    supported = coverage(
        products.stellar_grid,
        detector_pixels(products.stellar_grid),
        reconstruction_margin(stellar.reconstruction),
    )
    if speckle is not None:
        # The field owns its native lod grid, Frame, reconstruction and
        # extension; its clock subtracts its own epoch. The excess is signed
        # (coherent interference can lower the floor); only the total
        # leakage is clipped at zero below.
        excess = speckle.realize(wavelength_nm=wavelength_nm, time_d=time_d)
        leakage = leakage + onto_detector(excess)
        supported = supported & coverage(
            replace(excess.grid.to_arcsec(wavelength_nm), orientation=grid.orientation),
            grid,
            reconstruction_margin(excess.reconstruction),
        )
    leakage = gain * phi_star * jnp.maximum(leakage, 0.0)

    planet = jnp.zeros(grid.shape)
    if positions is not None:
        x, y = grid.orientation.sky_to_detector(positions[0], positions[1])
        phi = system.planet_flux_densities(
            wavelength_nm, jnp.atleast_1d(time_d), ephemeris=ephemeris
        )[:, 0]
        psfs = coro.source_psfs(
            wavelength_nm, x / lod_arcsec, y / lod_arcsec, grid=detector_pixels(native)
        )
        # A source outside the coronagraph's source domain has a zero image
        # and is flagged for that source alone: it contributes nothing and
        # leaves every other planet's image as it would be without it.
        samples = jnp.where(psfs.valid[:, None, None], psfs.image.data, 0.0)
        planet = gain * jnp.einsum("k,kyx->yx", phi, samples)

    local = jnp.zeros(grid.shape)
    if sky_radiance is not None:
        sky_map = coro.uniform_sky_map(
            wavelength_nm, grid=detector_pixels(products.sky_grid)
        )
        transmission = sky_map.data
        sky = Image(
            jnp.broadcast_to(gain * sky_radiance, grid.shape),
            grid,
            (),
            Measure.density("sr"),
            "ph s^-1 nm^-1",
        )
        local = sky.integrated().data * transmission
        supported = supported & coverage(
            products.sky_grid,
            detector_pixels(products.sky_grid),
            reconstruction_margin(sky_map.reconstruction),
        )

    disk = jnp.zeros(grid.shape)
    if system.disk is not None:
        # The scene operator places each source cell's flux at the cell centre,
        # so it needs the cell integral of the disk, not a midpoint sample.
        # Evaluated on the operator's own source grid (which reaches the edge
        # of its source domain, beyond the stored PSF grid), refined by an
        # integer factor and summed back to source cells, so sky off the
        # detector still scatters onto it.
        sources = coro.scene_grid
        fine = _refined(sources, _DISK_SUBSAMPLE)
        radiance = system.photon_radiance(
            wavelength_nm,
            time_d,
            grid=replace(fine.to_arcsec(wavelength_nm), orientation=grid.orientation),
        )
        scene = integrate_footprints(
            replace(radiance.integrated(), grid=fine), footprint_overlap(fine, sources)
        )
        rendered = coro.render_scene(scene, wavelength_nm, grid=detector_pixels(native))
        disk = gain * rendered.image.data
        supported = supported & coverage(native, detector_pixels(native))
    # The warm telescope radiates uniformly: no coronagraph factor, system
    # throughput, flux scale or sky transmission, only the pixel solid angle.
    thermal = (
        Image(
            jnp.broadcast_to(
                instrument.thermal_photon_rate_per_sr(wavelength_nm), grid.shape
            ),
            grid,
            (),
            Measure.density("sr"),
            "ph s^-1 nm^-1",
        )
        .integrated()
        .data
    )
    zero = jnp.zeros(grid.shape)
    components = dict(
        planet=planet,
        star_leakage=leakage,
        local_zodi=local,
        exozodi=disk,
        thermal=thermal,
        stellar_companions=zero,
    )
    return components, supported


def _band_ids(instrument, scene, quadrature):
    """Select, on the host, which coronagraph models the program needs.

    Band selection is structure, not a traced value: the program holds one
    branch per model the concrete nodes fall in (every model when the nodes are
    traced) and picks among them from the traced node wavelength.
    """
    coro = instrument.coronagraph
    models = coro.coronagraphs if isinstance(coro, MultiBandCoronagraph) else (coro,)
    nodes = quadrature.nodes_nm
    if isinstance(coro, MultiBandCoronagraph) and not isinstance(
        nodes, jax.core.Tracer
    ):
        chosen = [coro.coronagraph_at(float(w)) for w in np.unique(np.asarray(nodes))]
        ids = tuple(i for i, m in enumerate(models) if any(m is c for c in chosen))
    else:
        ids = tuple(range(len(models)))
    for i in ids:
        if not isinstance(models[i], AbstractImagingCoronagraph):
            raise TypeError("image formation requires an AbstractImagingCoronagraph")
        if scene.disk is not None and not isinstance(
            models[i], AbstractSceneCoronagraph
        ):
            raise TypeError("a system with a disk requires an AbstractSceneCoronagraph")
    return ids


def _node_function(models, scene, instrument, exposure, grid, planets, uniform_sky):
    """Return node(w): the rates and pixel support at one traced wavelength.

    Works for any model count; the result is ``(components, supported)``.

    With ``planets`` False no planet is propagated or rendered and the planet
    term is zero. The local zodi is the scene model's radiance for the scene's
    sky frame at the observation time; with ``uniform_sky`` it is one photon
    s^-1 m^-2 nm^-1 sr^-1 at every wavelength (still absent when the scene has
    no local zodi).
    """
    system, zodi = scene, scene.local_zodi
    time_d = jnp.asarray(exposure.time_d)

    def sky_radiance(w):
        if zodi is None:
            return None
        if uniform_sky:
            return jnp.ones_like(w)
        return zodi.photon_radiance(w, system.sky_frame, time_d)

    ephemeris, positions = None, None
    if planets and system.planets:
        times = jnp.atleast_1d(time_d)
        ephemeris = system.ephemeris(times)
        positions = system.positions(times, ephemeris=ephemeris)[:, :, 0]

    def with_model(model):
        return lambda w: _node_rates(
            w,
            model,
            system,
            instrument,
            time_d,
            grid,
            ephemeris,
            positions,
            sky_radiance(w),
        )

    if len(models) == 1:
        return with_model(models[0])

    def node(w):
        covered = jnp.stack(
            [(m.bandpass.low_nm <= w) & (w <= m.bandpass.high_nm) for m in models]
        )
        rates, supported = jax.lax.switch(
            jnp.argmax(covered), [with_model(m) for m in models], w
        )
        # A node outside every declared band has no performance model.
        return (
            jax.tree.map(lambda x: jnp.where(jnp.any(covered), x, jnp.nan), rates),
            supported,
        )

    return node


# Terms that read a coronagraph product. Thermal emission is uniform and reads
# none, so it stays where the products end (and the aperture record reads one
# pixel of it).
_PRODUCT_TERMS = ("planet", "star_leakage", "local_zodi", "exozodi")


def _zero_unsupported(values, supported):
    """Set the product-read terms to zero where ``supported`` is False.

    A band-integrated pixel is unsupported when any node of its band is, so the
    zero covers every node of the band and integrating the spectral record
    equals the streamed electrons.
    """
    # A NaN from a node outside every declared band is a different failure and
    # stays NaN.
    return {
        name: jnp.where(supported, x, jnp.where(jnp.isnan(x), x, 0.0))
        if name in _PRODUCT_TERMS
        else x
        for name, x in values.items()
    }


@eqx.filter_jit
def _render(scene, instrument, exposure, *, kind, band_ids, planets, uniform_sky):
    """Render every node of one observation as a single traced program.

    Returns the component images and the (W, ny, nx) pixel support, which
    :func:`_run` wraps in the Rates record.
    """
    coro = instrument.coronagraph
    models = (
        tuple(coro.coronagraphs[i] for i in band_ids)
        if isinstance(coro, MultiBandCoronagraph)
        else (coro,)
    )
    quadrature = exposure.quadrature
    grid = _detector_grid(instrument, SkyOrientation(exposure.roll_deg))
    node = _node_function(
        models, scene, instrument, exposure, grid, planets, uniform_sky
    )
    if kind == "photons":
        values, covered = jax.lax.map(node, quadrature.nodes_nm.ravel())
        shape = quadrature.nodes_nm.shape
        values = jax.tree.map(lambda x: x.reshape((*shape, *grid.shape)), values)
        supported = jnp.all(covered.reshape((*shape, *grid.shape)), axis=1)
        return _zero_unsupported(values, supported[:, None]), supported

    # Stream the nodes of each band: only its accumulated rate and one node
    # image are live.
    def render_band(i):
        row = quadrature.band(i)
        initial = {name: jnp.zeros(grid.shape) for name in Rates.component_names()}

        def step(carry, node_and_weight):
            total, supported = carry
            w, weight = node_and_weight
            components, covered = node(w)
            single = Quadrature(row.bands, w[None, None], weight[None, None])
            photons = Rates(
                single, "photon", **jax.tree.map(lambda x: x[None, None], components)
            )
            electrons = instrument.detector.electrons(photons)
            return (
                {name: a + getattr(electrons, name)[0] for name, a in total.items()},
                supported & covered,
            ), None

        xs = (row.nodes_nm[0], row.weights("response")[0])
        initial_support = jnp.ones(grid.shape, bool)
        total, supported = jax.lax.scan(step, (initial, initial_support), xs)[0]
        return _zero_unsupported(total, supported), supported

    rows = [render_band(i) for i in range(len(quadrature.bands))]
    return (
        jax.tree.map(lambda *x: jnp.stack(x), *(values for values, _ in rows)),
        jnp.stack([supported for _, supported in rows]),
    )


@eqx.filter_jit
def _draw(detector, key, electrons, plan):
    """Stage detector noise separately so observations share the renderer."""
    return detector.draw(key, electrons, plan)


def _run(scene, instrument, exposure, kind, *, planets=True, uniform_sky=False):
    """Run the shared program for one scene, instrument and exposure."""
    values, supported = _render(
        scene,
        instrument,
        exposure,
        kind=kind,
        band_ids=_band_ids(instrument, scene, exposure.quadrature),
        planets=planets,
        uniform_sky=uniform_sky,
    )
    quadrature = exposure.quadrature
    if kind == "photons":
        return Rates(quadrature, "photon", supported=supported, **values)
    return Rates(quadrature.bands, "electron", supported=supported, **values)


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

    Each band is split at the breaks the instrument and the scene's local zodi
    declare, and an ``order``-point rule is applied to every piece (four nodes
    per band when no break falls inside it). The star is left out of the
    breaks. ``roll_deg`` is the position angle of detector +y, north through
    east. ``plan`` is only needed by
    :func:`observe`. The local zodi is evaluated at render time for the scene's
    sky frame and ``time_d``.

    The build uses the scene only for the spectral breaks of its local zodi, a
    constant of the zodi model, so no scene value is cached in the result and it
    declares no read.

    Args:
        scene: The scene, used only for the breaks of its local zodi.
        instrument: The instrument, which declares its own spectral breaks.
        bands: The bands to observe, each split and integrated separately.
        time_d: Observation time in days from the scene's epoch.
        roll_deg: Position angle of detector +y, in degrees.
        plan: The frame plan, needed only by :func:`observe`. None for none.
        order: Number of quadrature nodes in each piece of a band.

    Returns:
        The exposure holding the time, roll, wavelength quadrature and plan.
    """
    models = [m for m in (instrument, scene.local_zodi) if m is not None]
    return Exposure(
        time_d=time_d,
        roll_deg=roll_deg,
        quadrature=Quadrature.build(bands, *models, order=order),
        plan=plan,
    )


def photon_rates(scene: Scene, instrument: Instrument, exposure: Exposure) -> Rates:
    """Return full (W,Q,ny,nx) photon integrands, in photons/s/nm/pixel.

    One entry per source term and wavelength node of ``exposure.quadrature``.
    Call integrate() on the returned record to integrate its spectral nodes.
    No QE or survival is applied. Local zodi comes from ``scene.local_zodi`` (None
    disables it); the speckle field from ``instrument.speckle`` (None adds
    none). Thermal is the warm telescope's uniform emission from
    ``instrument.warm_emission`` (zero when None); companion terms are zero.

    The coronagraph renders only what its data hold. ``supported`` of shape
    (W, ny, nx) is False at a detector pixel whose footprint, with the reach of
    each product's reconstruction, leaves any product the render reads at any
    node of the band; the planet, star leakage, local zodi and exozodi are zero
    there at every node. A planet outside the coronagraph's source domain
    contributes zero and leaves every other planet as it would be alone.

    Args:
        scene: The scene to render.
        instrument: The instrument to render it with.
        exposure: The exposure giving the time, roll and wavelength nodes.

    Returns:
        The photon integrands in photons/s/nm/pixel, with the ``supported``
        flag of shape (W, ny, nx).
    """
    return _run(scene, instrument, exposure, "photons")


def expected(scene: Scene, instrument: Instrument, exposure: Exposure) -> Rates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    These are the expected rates for the instantaneous scene at
    ``exposure.time_d``; ``exposure.plan`` does not enter. The exposure may be
    an argument of JIT, vmap and grad: node wavelengths, weights and roll are
    traced, and one program serves any of their values. Each scan step sends
    one photon image and its wavelength and weight to the detector, so forward
    storage is the band accumulator plus one node's images. JAX reverse-mode
    may retain residuals for differentiation.

    ``supported`` of shape (W, ny, nx) is False where any node of the band
    reads beyond a product's data (see :func:`photon_rates`); the band-integrated
    planet, star leakage, local zodi and exozodi are zero there.

    Args:
        scene: The scene to render.
        instrument: The instrument to render it with, including its detector.
        exposure: The exposure giving the time, roll and wavelength nodes.

    Returns:
        The expected rates in electrons/s, band integrated, shape (W, ny, nx),
        with the ``supported`` flag of shape (W, ny, nx).
    """
    return _run(scene, instrument, exposure, "electrons")


def observe(key, scene: Scene, instrument: Instrument, exposure: Exposure) -> Image:
    """Draw one electron Image (W,ny,nx), integrated on the detector sky grid.

    Rates describe the instantaneous scene at ``exposure.time_d``, held
    throughout ``exposure.plan``, which is required. Plan fields are scalar or
    (W,), with W matching the band axis. Detector noise is drawn once after
    combining all source terms.

    Args:
        key: JAX PRNG key for the detector noise draw.
        scene: The scene to render.
        instrument: The instrument to render it with, including its detector.
        exposure: The exposure giving the time, roll, wavelength nodes and the
            required frame plan.

    Returns:
        The noisy electron image of shape (W, ny, nx), one plane per band.
    """
    if exposure.plan is None:
        raise ValueError("observe needs an exposure built with a FramePlan")
    electrons = expected(scene, instrument, exposure)
    # A traced roll cannot live in static grid metadata, so the returned grid
    # then carries no orientation.
    roll = exposure.roll_deg
    orientation = None if isinstance(roll, jax.core.Tracer) else SkyOrientation(roll)
    return Image(
        _draw(instrument.detector, key, electrons, exposure.plan),
        _detector_grid(instrument, orientation),
        ("band",),
        Measure.integral(),
        "electrons",
    )
