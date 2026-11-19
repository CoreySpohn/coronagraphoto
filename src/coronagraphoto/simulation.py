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
from optixstuff.grids import Image, Measure, SkyOrientation, resample
from optixstuff.spectral import Band, Quadrature
from orbix.observatory import ecliptic_pointing
from skyscapes import Scene


def _detector_grid(instrument, orientation):
    """Construct the one sky grid shared by planets, disks and observations."""
    if orientation is not None and orientation.parity != -1:
        raise ValueError("detector pointing must have parity=-1")
    return replace(instrument.detector.grid, orientation=orientation)


def _star_sky_position(scene):
    """Right ascension and declination of the scene's star, in degrees (J2000).

    The one place this package reads where the target is on the sky.
    """
    return scene.star.ra_deg, scene.star.dec_deg


def _node_rates(
    wavelength_nm,
    coro,
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
    products = coro.at_wavelength(wavelength_nm)
    native = products.source_grid
    angular = replace(native.to_arcsec(wavelength_nm), orientation=grid.orientation)
    lod_arcsec = native.frame.lod_to_arcsec(wavelength_nm)
    gain = instrument.primary.collecting_area_m2 * instrument.system_throughput(
        wavelength_nm
    )

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
    if positions is not None:
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


def _band_ids(instrument, scene, quadrature):
    """Select, on the host, which coronagraph models the program needs.

    Band selection is structure, not a traced value: the program holds one
    branch per model the concrete nodes fall in (every model when the nodes are
    traced) and picks among them from the traced node wavelength.
    """
    coro = instrument.coronagraph
    models = coro.bands if isinstance(coro, MultiBandCoronagraph) else (coro,)
    nodes = quadrature.nodes_nm
    if isinstance(coro, MultiBandCoronagraph) and not isinstance(
        nodes, jax.core.Tracer
    ):
        chosen = [coro.band(float(w)) for w in np.unique(np.asarray(nodes))]
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


def _node_function(models, scene, instrument, exposure, grid, planets):
    """Return node(w): the rates at one traced wavelength, for any model count.

    With ``planets`` False no planet is propagated or rendered and the planet
    term is zero.
    """
    system, zodi = scene, scene.zodi
    time_d = jnp.asarray(exposure.time_d)
    # The local-zodi sightline is the target's, at the observation time.
    sightline = (
        ecliptic_pointing(*_star_sky_position(system), system.epoch, time_d)
        if zodi is not None
        else (None, None)
    )
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
            zodi,
            *sightline,
        )

    if len(models) == 1:
        return with_model(models[0])

    def node(w):
        covered = jnp.stack(
            [(m.bandpass.lo_nm <= w) & (w <= m.bandpass.hi_nm) for m in models]
        )
        rates = jax.lax.switch(jnp.argmax(covered), [with_model(m) for m in models], w)
        # A node outside every declared band has no performance model.
        return jax.tree.map(lambda x: jnp.where(jnp.any(covered), x, jnp.nan), rates)

    return node


@eqx.filter_jit
def _render(scene, instrument, exposure, *, kind, band_ids, planets):
    """Render every node of one observation as a single traced program."""
    coro = instrument.coronagraph
    models = (
        tuple(coro.bands[i] for i in band_ids)
        if isinstance(coro, MultiBandCoronagraph)
        else (coro,)
    )
    quadrature = exposure.quadrature
    grid = _detector_grid(instrument, SkyOrientation(exposure.roll_deg))
    node = _node_function(models, scene, instrument, exposure, grid, planets)
    if kind == "photons":
        values = jax.lax.map(node, quadrature.nodes_nm.ravel())
        return jax.tree.map(
            lambda x: x.reshape((*quadrature.nodes_nm.shape, *grid.shape)), values
        )

    # Stream the nodes of each band: only its accumulated rate and one node
    # image are live.
    def render_band(i):
        row = quadrature.band(i)
        initial = {name: jnp.zeros(grid.shape) for name in Rates.component_names()}

        def step(total, node_and_weight):
            w, weight = node_and_weight
            components = node(w)
            single = Quadrature(row.bands, w[None, None], weight[None, None])
            photons = Rates(
                single, "photon", **jax.tree.map(lambda x: x[None, None], components)
            )
            electrons = instrument.detector.electrons(photons)
            return {
                name: a + getattr(electrons, name)[0] for name, a in total.items()
            }, None

        xs = (row.nodes_nm[0], row.weights("response")[0])
        return jax.lax.scan(step, initial, xs)[0]

    rows = [render_band(i) for i in range(len(quadrature.bands))]
    return Rates(
        quadrature.bands, "electron", **jax.tree.map(lambda *x: jnp.stack(x), *rows)
    )


@eqx.filter_jit
def _draw(detector, key, electrons, plan):
    """Stage detector noise separately so observations share the renderer."""
    return detector.draw(key, electrons, plan)


def _run(scene, instrument, exposure, kind, *, planets=True):
    """Run the shared program for one scene, instrument and exposure."""
    return _render(
        scene,
        instrument,
        exposure,
        kind=kind,
        band_ids=_band_ids(instrument, scene, exposure.quadrature),
        planets=planets,
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
    east. ``plan`` is only needed by
    :func:`observe`. The local-zodi line of sight is derived at render time
    from the star's RA and Dec, the scene epoch and ``time_d``.
    """
    models = [m for m in (instrument, scene.zodi) if m is not None]
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
    No QE or survival is applied. Local zodi comes from ``scene.zodi`` (None
    disables it); the speckle field from ``instrument.speckle`` (None adds
    none). Thermal and companion terms are zero.
    """
    components = _run(scene, instrument, exposure, "photons")
    return Rates(exposure.quadrature, "photon", **components)


def expected(scene: Scene, instrument: Instrument, exposure: Exposure) -> Rates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    These are the expected rates for the instantaneous scene at
    ``exposure.time_d``; ``exposure.plan`` does not enter. The exposure may be
    an argument of JIT, vmap and grad: node wavelengths, weights and roll are
    traced, and one program serves any of their values. Each scan step sends
    one photon image and its wavelength and weight to the detector, so forward
    storage is the band accumulator plus one node's images. JAX reverse-mode
    may retain residuals for differentiation.
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
