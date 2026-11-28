"""Per-pixel readers of the shared photon and electron rate budgets.

Arrays are (band, node, y, x) for photons/s/nm and (band, y, x) for
band-integrated electrons/s. Detector shape, grid axes, frame and band
selection are static; the wavelength nodes and weights, the roll and the event
time are traced leaves, so one compiled program serves any node values and any
roll (a new node count or band count is a new array shape and compiles once).
The compiled stage sees leaves only: the commanded event and the sampling stay
on the host, which checks them against the observatory and translates the event
time to the scene's epoch and to the speckle field's own epoch, one clock each.
Scenes, observatories, events and samplings can all be arguments of JAX
transformations. For a band-averaged product, planet source images are read
once per observing band at its centre. Each node weights them by its planet
flux and gain, with detector terms applied per node for electrons.
Single-wavelength responses and the speckle field use the node wavelength.
The field size still sets the node count. For a band product, extra nodes
sample the spectra and speckles without refining the coronagraph geometry.
Scene positions, spectra, angular diameter and disk radiance enter instrument
calculations at the working precision of the band centers. The observatory,
the event and the sampling must have one uniform working precision.
A roll changes the position angle of detector +y, north through east.
"""

from collections.abc import Sequence
from dataclasses import replace
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.epoch import Epoch
from hwoutils.leaves import require_tree
from hwoutils.numerics import COARSEST
from hwoutils.precision import like, precision_of
from hwoutils.provenance import Digest, build_stamps, merge_stamps, qualify, stamp_value
from optixstuff import (
    Exposure,
    ExposureData,
    FramePlan,
    Instrument,
    Observatory,
    Pointing,
    Rates,
    SpectralSampling,
)
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
    resample,
)
from optixstuff.sampling import build_facts
from optixstuff.spectral import Band, Quadrature, Resolution, merge_tolerance_nm
from skyscapes import Scene

from coronagraphoto._version import __version__


def _detector_grid(sel, orientation):
    """Construct the one sky grid shared by planets, disks and observations."""
    if orientation is not None and orientation.parity != -1:
        raise ValueError("detector pointing must have parity=-1")
    return replace(sel.detector.grid, orientation=orientation)


def _exposure_grid(sel, pointing):
    """Read the detector grid at the commanded roll."""
    return _detector_grid(sel, SkyOrientation(pointing.roll_deg))


def _source_offsets_lod(wavelength_nm, pointing, native, positions):
    """Convert sky positions to detector offsets in the source product frame."""
    x, y = pointing.detector_offsets_arcsec(positions[0], positions[1])
    lod_arcsec = native.frame.lod_to_arcsec(wavelength_nm)
    return x / lod_arcsec, y / lod_arcsec


def _detector_pixels(grid, scale_nm, product_grid):
    """Express the detector pixels in the aligned axes of a product grid."""
    return replace(
        grid.to_lod(scale_nm, product_grid.frame), orientation=product_grid.orientation
    )


def _planet_samples(coro, wavelength_nm, scale_nm, grid, pointing, native, positions):
    """Read all planet images, zeroing each source outside the product domain."""
    x, y = _source_offsets_lod(scale_nm, pointing, native, positions)
    psfs = coro.source_psfs(
        wavelength_nm, x, y, grid=_detector_pixels(grid, scale_nm, native)
    )
    return jnp.where(psfs.valid[:, None, None], psfs.image.data, 0)


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
    band_center_nm,
    coro,
    system,
    sel,
    time_d,
    speckle_time_d,
    grid,
    pointing,
    positions,
    node,
    samples,
):
    """Render the photon rates at one node.

    For a band-averaged product, ``samples`` holds the planet source images
    read once per observing band at its centre. This node weights the images
    by its planet flux and gain. The electron reader applies its detector terms.

    ``node`` holds this node's row of ``Scene.sources``: ``star`` (the
    node-effective stellar photon flux density, with its scale), ``planets``
    (one node-effective photon flux density per planet), ``sky_radiance`` (the
    local-zodi radiance in photons s^-1 m^-2 nm^-1 sr^-1, or None for no local
    zodi), ``diameter`` (stellar angular diameter in arcsec) and ``node_star``,
    the star the disk is rendered against. Returns the
    component images and two booleans (ny, nx) of detector pixels covered at
    this node: the static coverage, the AND of ``coverage`` over every
    coronagraph product the render reads, each with the reach of the read
    ``resample`` performs, and the coverage of the speckle field's grid (all
    True without a field). Nothing is reflected or extrapolated, so the light
    at an uncovered pixel is not modeled; the caller zeroes it.
    """
    speckle = sel.speckle
    # A band average already includes the chromatic blur across its band.
    scale_nm = coro.scale_wavelength_nm(wavelength_nm, band_center_nm)
    products = coro.at_wavelength(wavelength_nm)
    native = products.source_grid
    gain = sel.effective_area_m2(wavelength_nm)

    detector_pixels = partial(_detector_pixels, grid, scale_nm)

    def onto_detector(image):
        # Native coordinates describe detector axes, so attach the live pointing
        # only after the coronagraph has rendered its instrumental response.
        product_grid = replace(
            image.grid.to_arcsec(wavelength_nm), orientation=grid.orientation
        )
        return resample(replace(image, grid=product_grid), grid).data

    phi_star = node["star"]
    diameter = node["diameter"]
    stellar = coro.stellar_map(
        wavelength_nm,
        diameter / products.stellar_grid.frame.lod_to_arcsec(scale_nm),
        grid=detector_pixels(products.stellar_grid),
    )
    leakage = stellar.data
    speckle_covered = jnp.ones(grid.shape, bool)
    supported = coverage(
        products.stellar_grid,
        detector_pixels(products.stellar_grid),
        reconstruction=stellar.reconstruction,
        measure=stellar.measure,
    )
    if speckle is not None:
        # The field owns its native lod grid, Frame and reconstruction, and
        # ends where its data end: its grid, narrowed by its data mask when
        # its array is wider than its data. The event time is converted by the
        # field's ``elapsed_s`` from the event stated on the field's own epoch
        # (``speckle_time_d``), which refuses a float32 time at a nonzero
        # epoch. Its coverage is kept apart from the static products'. The
        # excess is signed (coherent interference can lower the floor); only
        # the total leakage is clipped at zero below.
        excess = speckle.realize(
            wavelength_nm=wavelength_nm, elapsed_s=speckle.elapsed_s(speckle_time_d)
        )
        leakage = leakage + onto_detector(excess)
        speckle_covered = coverage(
            replace(excess.grid.to_arcsec(wavelength_nm), orientation=grid.orientation),
            grid,
            reconstruction=excess.reconstruction,
            measure=excess.measure,
            valid=speckle.data_mask(),
        )
    # A guard on the star plus the signed speckle excess: a star read alone is
    # positive wherever it is supported, so this binds only on an excess.
    leakage = gain * phi_star * jnp.maximum(leakage, 0)

    zero = jnp.zeros_like(leakage)
    planet = zero
    if positions is not None:
        if samples is None:
            samples = _planet_samples(
                coro, wavelength_nm, scale_nm, grid, pointing, native, positions
            )
        planet = gain * jnp.einsum("k,kyx->yx", node["planets"], samples)

    local = zero
    sky_radiance = node["sky_radiance"]
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
            reconstruction=sky_map.reconstruction,
            measure=sky_map.measure,
        )

    disk = zero
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
            grid=replace(fine.to_arcsec(scale_nm), orientation=grid.orientation),
            star=node["node_star"],
        )
        radiance = replace(radiance, data=like(radiance.data, band_center_nm))
        scene = integrate_footprints(
            replace(radiance.integrated(), grid=fine), footprint_overlap(fine, sources)
        )
        rendered = coro.render_scene(scene, wavelength_nm, grid=detector_pixels(native))
        disk = gain * rendered.image.data
        supported = supported & coverage(native, detector_pixels(native))
    # The warm telescope radiates uniformly: no coronagraph factor, system
    # throughput, flux scale or sky transmission, only the pixel solid angle.
    # A native cold instrument needs no radiance read. Honor custom models.
    thermal = zero
    if (
        sel.warm_emission is not None
        or type(sel.instrument).thermal_photon_rate_per_sr
        is not Instrument.thermal_photon_rate_per_sr
    ):
        thermal = (
            Image(
                jnp.broadcast_to(
                    sel.thermal_photon_rate_per_sr(wavelength_nm), grid.shape
                ),
                grid,
                (),
                Measure.density("sr"),
                "ph s^-1 nm^-1",
            )
            .integrated()
            .data
        )
    components = dict(
        planet=planet,
        star_leakage=leakage,
        local_zodi=local,
        exozodi=disk,
        thermal=thermal,
        stellar_companions=zero,
    )
    return components, supported, speckle_covered


def _band_ids(coro, scene, quadrature):
    """Select, on the host, which coronagraph models the program needs.

    For averaged products, return the covering child index for each observing
    band. For single-wavelength products, return the children the real nodes
    can select, in dispatch order (all children when the nodes are traced).
    """
    models = coro.coronagraphs if isinstance(coro, MultiBandCoronagraph) else (coro,)
    nodes = quadrature.nodes_nm
    if isinstance(coro, MultiBandCoronagraph) and coro.band_averaged:
        # One branch per observing band, even when child bandpasses overlap.
        indices = {id(model): i for i, model in enumerate(models)}
        ids = tuple(
            indices[id(coro.coronagraph_for_band(band))] for band in quadrature.bands
        )
    elif isinstance(coro, MultiBandCoronagraph) and not isinstance(
        nodes, jax.core.Tracer
    ):
        ids = _node_child_ids(models, quadrature)
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


def _identity(fn):
    return fn


def _planet_positions(system, time_d, ref):
    """Read sky positions and retain the ephemeris for other source reads."""
    times = jnp.atleast_1d(time_d)
    ephemeris = system.ephemeris(times)
    return ephemeris, like(system.positions(times, ephemeris=ephemeris), ref)


def _covering_models(models, wavelength_nm, xp):
    """Compare inclusive band edges on the common float32 wavelength grid.

    Expand the host edges by the spectral merge tolerance. Comparing at the
    coarsest precision lets concrete and traced nodes make the same choice.
    The original wavelength still supplies the radiometric read.
    """
    edges = np.array(
        [(m.bandpass.low_nm, m.bandpass.high_nm) for m in models], dtype=np.float64
    )
    direction = np.array([1, -1], dtype=COARSEST)
    expanded = edges - direction * merge_tolerance_nm(edges)
    limits = expanded.astype(COARSEST)
    # Round each threshold inward so the common grid adds no extra slack.
    limits = np.where(
        direction * (limits - expanded) < 0,
        np.nextafter(limits, direction * np.inf),
        limits,
    )
    wave = xp.asarray(wavelength_nm, dtype=COARSEST)
    return (limits[:, 0] <= wave) & (wave <= limits[:, 1])


def _host_child_index(models, wavelength_nm):
    """Select the first child covering a concrete node or band center."""
    covered = np.flatnonzero(_covering_models(models, wavelength_nm, np))
    if not covered.size:
        raise ValueError(f"No coronagraph bandpass covers {wavelength_nm} nm")
    return int(covered[0])


def _node_child_ids(models, quadrature):
    """Select real nodes' children on the host, in dispatch order."""
    rows = np.asarray(quadrature.nodes_nm)
    real = np.concatenate(
        [row[:n] for row, n in zip(rows, _real_counts(quadrature), strict=True)]
    )
    selected = {_host_child_index(models, w) for w in np.unique(real)}
    return tuple(i for i in range(len(models)) if i in selected)


def _band_center_dtype(quadrature, sel):
    """Floating working precision for a band center, even with integer nodes."""
    floating = (
        values
        for values in (quadrature.nodes_nm, quadrature.weights_nm)
        if jnp.issubdtype(values.dtype, jnp.floating)
    )
    return jnp.result_type(sel.flux_scale, *floating)


def _band_centers(quadrature, sel):
    """Band centers in the floating working precision, including for integer nodes."""
    return jnp.asarray(
        [band.center_nm for band in quadrature.bands],
        dtype=_band_center_dtype(quadrature, sel),
    )


def _node_function(
    models,
    scene,
    sel,
    quadrature,
    resolution,
    pointing,
    grid,
    ephemeris,
    positions,
    time_d,
    speckle_time_d,
    centers,
    uniform_sky,
    band_branches,
):
    """Return the checkpointed function for one node's rates and pixel support.

    ``w`` is the traced node wavelength and ``(i, j)`` its band and node index
    in the sampling's quadrature. The sources are read once, on the sampling's
    quadrature and resolution, and each node takes its row. Works for any model
    count; the result is ``(components, static, speckle)``, the static and
    speckle coverage of :func:`_node_rates`.

    A band-averaged product's planet source images are read once per observing
    band at its centre and passed into the checkpoint. Each node weights them
    by its flux and gain, with detector terms applied per node for electrons.
    With no planet positions the planet term is zero. The local zodi is the
    scene model's node-effective radiance for its sky frame at observation
    time. With ``uniform_sky`` it is one photon s^-1 m^-2 nm^-1 sr^-1 at every
    wavelength (still absent when the scene has no local zodi).

    The returned function is wrapped in :func:`jax.checkpoint`, so reverse-mode
    differentiation recomputes one node on the backward pass instead of
    holding every node's intermediates of the coronagraph synthesis. Forward
    evaluation and forward-mode differentiation are unchanged. Under
    ``jax.disable_jit()`` the wrapper is skipped, because a checkpoint traces its
    body even eagerly and eager debugging then loses concrete values.
    """
    system, zodi = scene, scene.local_zodi

    # A uniform sky reads no zodi, so a model that needs a sightline is not
    # asked for one.
    read = (
        eqx.tree_at(lambda s: s.local_zodi, system, None, is_leaf=lambda x: x is None)
        if uniform_sky
        else system
    )
    sources = read.sources(
        quadrature,
        resolution,
        jnp.atleast_1d(time_d),
        ephemeris=ephemeris,
    )
    star_nodes = like(sources.star[..., 0], centers)
    planet_nodes = like(sources.planets[..., 0], centers)
    zodi_nodes = None
    if zodi is not None and not uniform_sky:
        zodi_nodes = like(sources.local_zodi[..., 0], centers)
    diameter = like(sources.star_diameter_arcsec, centers)

    def row(w, i, j):
        if zodi is None:
            sky = None
        elif uniform_sky:
            sky = jnp.ones_like(w, dtype=centers.dtype)
        else:
            sky = zodi_nodes[i, j]
        return dict(
            star=star_nodes[i, j],
            planets=planet_nodes[i, j],
            sky_radiance=sky,
            diameter=diameter,
            node_star=sources.node_star(i, j) if system.disk is not None else None,
        )

    def with_model(model):
        return lambda w, i, j, samples=None: _node_rates(
            w,
            centers[i],
            model,
            system,
            sel,
            time_d,
            speckle_time_d,
            grid,
            pointing,
            positions,
            row(w, i, j),
            samples,
        )

    remat = _identity if jax.config.jax_disable_jit else jax.checkpoint

    if len(models) == 1:
        return remat(with_model(models[0]))

    branches = [with_model(m) for m in models]
    if band_branches is not None:
        indices = jnp.asarray(band_branches)
        return remat(
            lambda w, i, j, samples=None: jax.lax.switch(
                indices[i], branches, w, i, j, samples
            )
        )

    def node(w, i, j):
        covering = _covering_models(models, w, jnp)
        covered = jnp.any(covering)
        rates, static, speckle = jax.lax.switch(jnp.argmax(covering), branches, w, i, j)
        # A node outside every declared band has no performance model.
        return (
            jax.tree.map(lambda x: jnp.where(covered, x, x.dtype.type(jnp.nan)), rates),
            static,
            speckle,
        )

    return remat(node)


def _planet_support(model, pointing, center, positions):
    """Read one band's source-domain mask on the detector orientation."""
    native = model.at_wavelength(center).source_grid
    x, y = _source_offsets_lod(center, pointing, native, positions)
    return model.source_psfs(center, x.ravel(), y.ravel()).valid.reshape(
        positions.shape[1:]
    )


def planet_source_support(
    scene: Scene,
    observatory: Observatory,
    exposure: Exposure,
    *,
    sampling: SpectralSampling,
) -> np.ndarray:
    """Read whether each planet is inside the coronagraph's source domain.

    Entry ``[w, k, t]`` is True where planet ``k`` at time ``t`` lies inside
    the source domain of the coronagraph component the renderer selects for
    band ``w``: ``source_psfs(...).valid`` of that component, read at the band
    center with the renderer's planet positions, roll and lambda/D conversion.
    For a band average the renderer selects a component covering the whole
    band. For single-wavelength products, the sole component held for the
    real nodes is used directly. When several are held, the first component
    covering the center is selected using the spectral merge tolerance.
    For single-wavelength products the renderer admits each planet at each
    node: the separation in lambda/D changes across a band, and if it crosses a
    component boundary the nodes on either side read different components. A
    False flag therefore does not guarantee a zero band image, nor a True flag a
    nonzero one, when admission changes inside the band. Times are
    ``exposure.on_epoch(scene.epoch).time_d`` (a scalar gives T = 1).
    This is a host function on concrete values. It checks the event against
    the sampling and renders nothing.

    Examples:
        Two bands, each read through the component that covers its center::

            bands = (Band(520.0, 530.0), Band(570.0, 580.0))
            exposure = build_exposure(
                bands, epoch=scene.epoch, instrument="imager", time_d=0.0
            )
            sampling = build_sampling(observatory, exposure, scene=scene)
            support = planet_source_support(
                scene, observatory, exposure, sampling=sampling
            )
            support.shape  # (2, number of planets, 1)
            support[:, 0, 0]  # e.g. [False, True]: outside the blue grid only

    Args:
        scene: The unbatched scene supplying planet positions.
        observatory: The observatory supplying the coronagraph and detector grid.
        exposure: The commanded event supplying the bands, times and roll.
        sampling: The wavelength rule built for this event.

    Returns:
        A NumPy Boolean array of shape (W, K, T), one entry per band, planet
        and time. A planet-free scene has K equal to zero.

    Raises:
        NotImplementedError: The exposure has an explicit pointing offset.
        ValueError: A band center falls outside every multiband component the
            renderer holds for the quadrature, or the instrument is not named
            in the observatory.
        StampMismatchError: The sampling was built for other hardware.
    """
    sel, coro, time_d, _ = _host_stage(scene, observatory, exposure, sampling)
    quadrature = sampling.quadrature
    centers = _band_centers(quadrature, sel)
    _, positions = _planet_positions(scene, like(time_d, centers), centers)
    _, n_planets, n_times = positions.shape
    support = np.zeros((len(quadrature.bands), n_planets, n_times), dtype=bool)
    if not n_planets:
        return support
    # The components the renderer holds for this quadrature, in dispatch order.
    models = (
        tuple(coro.coronagraphs[i] for i in _band_ids(coro, scene, quadrature))
        if isinstance(coro, MultiBandCoronagraph)
        else (coro,)
    )
    for w, center in enumerate(centers):
        # The renderer reads a sole held component at every node, as here.
        model = models[0]
        if model.band_averaged and len(models) > 1:
            model = models[w]
        elif len(models) > 1:
            model = models[_host_child_index(models, quadrature.bands[w].center_nm)]
        support[w] = np.asarray(
            _planet_support(model, exposure.pointing, center, positions)
        )
    return support


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
        name: jnp.where(supported, x, jnp.where(jnp.isnan(x), x, 0))
        if name in _PRODUCT_TERMS
        else x
        for name, x in values.items()
    }


def _real_counts(quadrature):
    """Real nodes of each band from the static layout; every node if unknown."""
    if quadrature.pieces is None:
        return (quadrature.nodes_nm.shape[1],) * len(quadrature.bands)
    return tuple(sum(n for *_, n in band) for band in quadrature.pieces)


def _real_nodes(quadrature):
    """Band index, node index and padded-slot source of the real nodes, as host arrays.

    The first two list every real node in band order. ``source`` has the
    quadrature's (W, Q) shape, flattened, and gives for each slot the position of
    its node in that list; a padded slot points at the first real node of its
    band.
    """
    counts = _real_counts(quadrature)
    q = quadrature.nodes_nm.shape[1]
    bands = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    nodes = np.concatenate([np.arange(n) for n in counts])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.where(
        np.arange(q)[None, :] < np.asarray(counts)[:, None], np.arange(q), 0
    )
    source = (starts[:, None] + slot).ravel()
    return jnp.asarray(bands), jnp.asarray(nodes), jnp.asarray(source)


@eqx.filter_jit
def _render(
    scene,
    sel,
    coro,
    pointing,
    time_d,
    speckle_time_d,
    quadrature,
    resolution,
    *,
    kind,
    band_ids,
    planets,
    uniform_sky,
    static_beyond_speckle=False,
):
    """Render every node of one observation as a single traced program.

    Returns the component images and the (W, ny, nx) static and speckle
    support, which :func:`_run` wraps in the Rates record. The product terms
    are zeroed where the strict support (both) is False, or only where the
    static support is False when ``static_beyond_speckle``.
    """
    models = (coro,)
    band_branches = None
    if isinstance(coro, MultiBandCoronagraph):
        # Several observing bands can use the same averaged product. Trace
        # its renderer only once and select that branch for each of its bands.
        model_ids = tuple(dict.fromkeys(band_ids))
        models = tuple(coro.coronagraphs[i] for i in model_ids)
        if coro.band_averaged:
            band_branches = tuple(model_ids.index(i) for i in band_ids)
    grid = _exposure_grid(sel, pointing)
    centers = _band_centers(quadrature, sel)
    time_d = like(time_d, centers)
    if speckle_time_d is not None:
        speckle_time_d = like(speckle_time_d, centers)
    ephemeris, positions = None, None
    if planets and scene.planets:
        ephemeris, positions = _planet_positions(scene, time_d, centers)
        positions = positions[:, :, 0]

    node = _node_function(
        models,
        scene,
        sel,
        quadrature,
        resolution,
        pointing,
        grid,
        ephemeris,
        positions,
        time_d,
        speckle_time_d,
        centers,
        uniform_sky,
        band_branches,
    )
    reuse_samples = positions is not None and coro.band_averaged

    def band_samples(i):
        model = models[0 if band_branches is None else band_branches[i]]
        center = centers[i]
        native = model.at_wavelength(center).source_grid
        scale_nm = model.scale_wavelength_nm(center, center)
        return _planet_samples(
            model, center, scale_nm, grid, pointing, native, positions
        )

    if kind == "photons":
        shape = quadrature.nodes_nm.shape
        bands, nodes, source = _real_nodes(quadrature)
        per_band = [band_samples(i) for i in range(shape[0])] if reuse_samples else []
        # Stacking promotes every band to one dtype, so children that read in
        # different dtypes keep the per-node read here.
        samples = (
            jnp.stack(per_band)
            if per_band and len({s.dtype for s in per_band}) == 1
            else None
        )

        def mapped(indices):
            w, i, j = indices
            if samples is None:
                return node(w, i, j)
            return node(w, i, j, samples[i])

        values, static, speckle = jax.lax.map(
            mapped,
            (quadrature.nodes_nm[bands, nodes], bands, nodes),
        )
        # Each padded slot repeats the first real node of its band.
        values = jax.tree.map(
            lambda x: x[source].reshape((*shape, *grid.shape)), values
        )
        static, speckle = (
            jnp.all(c[source].reshape((*shape, *grid.shape)), axis=1)
            for c in (static, speckle)
        )
        kept = static if static_beyond_speckle else static & speckle
        return _zero_unsupported(values, kept[:, None]), static, speckle

    # Stream each band with its accumulated rate, shared PSFs and one node image.
    def render_band(i):
        row = quadrature.band(i)
        band_node = partial(node, samples=band_samples(i)) if reuse_samples else node

        def read_node(node_and_weight):
            w, weight, j = node_and_weight
            components, static_node, speckle_node = band_node(w, i, j)
            single = Quadrature(row.bands, w[None, None], weight[None, None])
            photons = Rates(
                single, "photon", **jax.tree.map(lambda x: x[None, None], components)
            )
            electrons = sel.detector.electrons(photons)
            values = {name: getattr(electrons, name)[0] for name in components}
            return values, static_node, speckle_node

        def step(carry, node_and_weight):
            total, static, speckle = carry
            values, static_node, speckle_node = read_node(node_and_weight)
            return (
                jax.tree.map(lambda a, b: a + b, total, values),
                static & static_node,
                speckle & speckle_node,
            ), None

        real = _real_counts(quadrature)[i]
        xs = (row.nodes_nm[0, :real], row.weights()[0, :real], jnp.arange(real))
        # The detector result sets each carry's dtype. Keep every node in the
        # same scan so the reduction order and checkpointing stay the same.
        first = jax.tree.map(lambda x: x[0], xs)
        template, _, _ = (
            read_node(first)
            if jax.config.jax_disable_jit
            else jax.eval_shape(read_node, first)
        )
        initial = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), template)
        everywhere = jnp.ones(grid.shape, bool)
        total, static, speckle = jax.lax.scan(
            step, (initial, everywhere, everywhere), xs
        )[0]
        kept = static if static_beyond_speckle else static & speckle
        return _zero_unsupported(total, kept), static, speckle

    rows = [render_band(i) for i in range(len(quadrature.bands))]
    return (
        jax.tree.map(lambda *x: jnp.stack(x), *(row[0] for row in rows)),
        jnp.stack([row[1] for row in rows]),
        jnp.stack([row[2] for row in rows]),
    )


@eqx.filter_jit
def _draw(detector, key, electrons, plan):
    """Stage detector noise separately so observations share the renderer."""
    return detector.draw(key, electrons, plan)


def _run(
    scene,
    sel,
    coro,
    exposure,
    clocks,
    sampling,
    kind,
    *,
    planets=True,
    uniform_sky=False,
    static_beyond_speckle=False,
):
    """Run the shared program for one scene, selected instrument and event.

    ``clocks`` is the pair ``(time_d, speckle_time_d)`` stated on the scene's
    epoch and on the speckle field's epoch. The compiled stage receives leaves
    and structure only, never the event or the sampling object.

    The record always carries both flags: ``supported`` is the static support
    AND the speckle field's, and ``static_supported`` the static support alone
    (the two are equal without a field).
    """
    quadrature = sampling.quadrature
    values, static, speckle = _render(
        scene,
        sel,
        coro,
        exposure.pointing,
        *clocks,
        quadrature,
        sampling.resolution,
        kind=kind,
        band_ids=_band_ids(coro, scene, quadrature),
        planets=planets,
        uniform_sky=uniform_sky,
        # Structure, not a value: a NumPy or JAX boolean is read on the host so
        # that filter_jit does not trace it.
        static_beyond_speckle=bool(static_beyond_speckle),
    )
    axis, carrier = (
        (quadrature, "photon") if kind == "photons" else (quadrature.bands, "electron")
    )
    return Rates(
        axis,
        carrier,
        supported=static & speckle,
        static_supported=static,
        **values,
    )


def _resolve(scene, quadrature):
    """Build the Resolution of a scene's tables on ``quadrature``, on the host.

    The tables are resolved on the exposure's own quadrature. A disk calibrated
    under the bolometric luminosity rule averages the star over a band, so each
    such band gets a quadrature of its own, resolved for the star's table only:
    the other tables need no rule on a band they do not cover.
    """
    tables = scene.spectral_tables()
    resolution = Resolution.build(quadrature, *tables)
    star = scene.star
    if not any(table is star for table in tables):
        return resolution
    quadratures, keys, rules = (
        list(resolution.quadratures),
        list(resolution.keys),
        list(resolution.rules),
    )
    for band in scene.calibration_bands():
        extra = Quadrature.build((band,))
        added = Resolution.build(extra, star)
        quadratures.append(extra)
        for key, rule in zip(added.keys, added.rules, strict=True):
            if key not in keys:
                keys.append(key)
                rules.append(rule)
    return Resolution(tuple(quadratures), tuple(keys), tuple(rules))


def _refuse_numpy_leaves(scene):
    """Raise ``TypeError`` naming every NumPy leaf of the scene.

    A NumPy leaf is the one violation that can stand in for a value the host
    build depends on, so it is refused here with its path, by the same walk as
    ``require_tree`` (it enters a user's own Module nested in the scene). The
    other leaf violations of the rule (weak scalars, Python numbers, a float
    dtype that is not the active one) are still produced by the libraries that
    build scenes and are left to those constructors; widen this to
    ``require_tree`` once they convert every field. Tracers are never NumPy
    leaves, so a scene under ``jit`` passes.
    """
    try:
        require_tree(scene)
    except TypeError as err:
        lines = [x for x in str(err).splitlines() if x.endswith("(NumPy leaf)")]
        if lines:
            raise TypeError(
                "the scene holds NumPy leaves; build it through its constructors, or "
                "pass replacement values through as_float_leaf or as_leaf:\n"
                + "\n".join(lines)
            ) from None


def _require_on_axis(exposure):
    """Reject explicit pointing offsets before reading any exposure data."""
    if exposure.pointing.offset_mas is not None:
        raise NotImplementedError(
            "coronagraphoto does not yet render pointing offsets; use offset_mas=None"
        )


def _check_render_inputs(scene, observatory, exposure, sampling):
    """Require on-axis pointing and check render precision and scene leaves."""
    _require_on_axis(exposure)
    precision_of(observatory)
    precision_of(exposure)
    precision_of(sampling)
    _refuse_numpy_leaves(scene)


def _host_stage(scene, observatory, exposure, sampling):
    """Run the host checks and translations that precede every render.

    Selects the configuration, binds its coronagraph once, checks the event
    against the sampling, states the event time on the scene's epoch and on the
    speckle field's own epoch (in binary64 on the host, before any precision
    cast), then refuses an explicit offset and mixed precision.

    Returns:
        ``(sel, coronagraph, time_d, speckle_time_d)``; the last is None
        without a speckle field.
    """
    sel = observatory.select(exposure.instrument)
    coro = sel.coronagraph
    sampling.check_event(exposure, selected=sel, scene=scene)
    time_d = exposure.on_epoch(scene.epoch).time_d
    field = sel.speckle
    speckle_time_d = None if field is None else exposure.on_epoch(field.epoch).time_d
    _check_render_inputs(scene, observatory, exposure, sampling)
    return sel, coro, time_d, speckle_time_d


def build_exposure(
    bands: Sequence[Band],
    *,
    epoch: Epoch,
    instrument: str,
    time_d,
    roll_deg=0.0,
    offset_mas=None,
    plan: FramePlan | None = None,
) -> Exposure:
    """Build the commanded event of one observation.

    The event holds no scene, no observatory and no wavelength rule; it is what
    a scheduler commands. Build the rule that serves it with
    :func:`build_sampling`. ``roll_deg`` is the position angle of detector +y,
    north through east. ``plan`` is only needed by :func:`observe`. Nothing is
    checked against hardware here: an explicit offset is refused when the event
    is rendered.

    Args:
        bands: The bands to observe, in order.
        epoch: Day zero of ``time_d``.
        instrument: Name of the instrument configuration.
        time_d: Start of collection in days from ``epoch``.
        roll_deg: Position angle of detector +y, in degrees.
        offset_mas: Target position from the optical axis in detector (x, y)
            milliarcseconds. Rendering requires None (on axis).
        plan: The frame plan, needed only by :func:`observe`. None for none.

    Returns:
        The commanded event.

    Raises:
        ValueError: An explicit pointing offset has a shape other than (2,).
    """
    return Exposure(
        time_d,
        Pointing(roll_deg, offset_mas),
        plan,
        epoch=epoch,
        instrument=instrument,
        bands=tuple(bands),
    )


def _scene_reads(scene, resolution):
    """Owner-qualified scene paths of the tables whose operators the build froze."""
    if not resolution.keys:
        return ()
    paths = []

    def tabulated(source):
        return getattr(source, "knot_digest", None) is not None

    if tabulated(scene.star):
        paths.append("star")
    for i, planet in enumerate(scene.planets):
        if tabulated(planet.physical_model):
            paths += [f"planets[{i}].physical_model", "star"]
    if tabulated(scene.local_zodi):
        paths.append("local_zodi")
    return qualify("scene", dict.fromkeys(paths))


def build_sampling(
    observatory: Observatory,
    exposure: Exposure,
    *,
    scene: Scene | None = None,
    order: int = 4,
    field_lod=None,
    allow_wide_bands: bool = False,
) -> SpectralSampling:
    """Build the wavelength rule and operators that serve one commanded event.

    Each band of the event is split at the breaks the selected instrument (its
    telescope included), the scene's local zodi and its disk declare, and a
    Gauss-Legendre rule is applied to every piece: ``order`` nodes, raised for a
    piece the diffraction pattern slides across by the detector field
    (``field_lod``). The field sets the node count for both product types. For a
    band-averaged product, every node uses the observing band's central
    lambda/D scale. Extra nodes sample the source spectra and speckle field, and
    do not refine the coronagraph image. For a single-wavelength product the
    diffraction pattern scales at every node. The star, the planets and a
    tabulated local zodi supply no breaks. The sampling holds the
    ``Resolution`` that reduces each tabulated spectrum on the nodes, so a band
    integral of a rough stellar spectrum is exact in the spectrum however few
    nodes the band has. A disk has no such reduction and is read at the nodes,
    so a disk with a tabulated spectrum (an ExoVista contrast cube) declares its
    wavelength knots as breaks and every knot interval inside a band gets its
    own nodes; a disk that declares no breaks adds none.

    The build reads the static knot grids of the scene's tables. When the
    resolution holds an operator, ``reads`` names the frozen node rule
    (``sampling.quadrature.nodes_nm`` and ``weights_nm``) and the scene tables
    the operators were built from; a smooth scene declares none. ``built_at``
    records the host facts that determined the sampling, so the renderer can
    check that it serves the event and the hardware it is given.

    Args:
        observatory: The observatory, whose selected instrument declares its own
            spectral breaks.
        exposure: The commanded event, which names the bands and the instrument.
        scene: The scene, whose tabulated sources are resolved on the nodes, or
            None to build from the hardware and the event alone. Without a
            scene the sampling adds no scene breaks, holds no resolution, reads
            no scene tables and carries no scene-layout stamp, so the check
            reports the scene layout as unverified.
        order: Number of quadrature nodes in each piece of a band.
        field_lod: Largest separation, in lambda/D at the band center, at which
            the response is read: a number, one value per band, or None for the
            field the instrument declares, its detector half-diagonal. Pieces
            the diffraction pattern slides across get more nodes, so a caller
            that reads a smaller field passes it here and pays for fewer
            renders; zero gives every piece ``order`` nodes.
        allow_wide_bands: Allow an observing band wider in fractional width
            than 1.25 times a band-averaged product. As in the YIP count rate
            recipe, the image then has only the product's blur at one scale and
            is under-blurred for the wider band.

    Returns:
        The quadrature, its resolution, the frozen reads and the build stamps.

    Raises:
        ValueError: The instrument is not named in the observatory, or an
            observing band exceeds the coverage of a tabulated source.
        TypeError: The scene contains NumPy leaves.
    """
    sel = observatory.select(exposure.instrument)
    if scene is None:
        models = [sel]
        disk_breaks = ()
    else:
        _refuse_numpy_leaves(scene)
        models = [m for m in (sel, scene.local_zodi) if m is not None]
        # The disk supplies breaks only: its knots split the band, and it has no
        # say in the field that sets the node count of a piece.
        disk_breaks = () if scene.disk is None else scene.disk.spectral_breaks_nm()
    quadrature = Quadrature.build(
        exposure.bands,
        *models,
        order=order,
        breaks_nm=disk_breaks,
        field_lod=field_lod,
        allow_wide_bands=allow_wide_bands,
    )
    if scene is None:
        resolution = Resolution.build(quadrature)
    else:
        resolution = _resolve(scene, quadrature)
    reads = ()
    if resolution.keys:
        reads = qualify(
            "sampling", ("quadrature.nodes_nm", "quadrature.weights_nm")
        ) + _scene_reads(scene, resolution)
    built_at = build_facts(
        exposure.bands,
        instrument=exposure.instrument,
        selected=sel,
        scene=scene,
        field_lod=field_lod,
        order=order,
        allow_wide_bands=allow_wide_bands,
        layout=Digest(quadrature.digest),
        precision=str(precision_of(observatory)),
        producer="coronagraphoto.build_sampling",
        producer_version=__version__,
    )
    return SpectralSampling(quadrature, resolution, reads=reads, built_at=built_at)


def photon_rates(
    scene: Scene,
    observatory: Observatory,
    exposure: Exposure,
    *,
    sampling: SpectralSampling,
    static_beyond_speckle: bool = False,
) -> Rates:
    """Return full (W,Q,ny,nx) photon integrands, in photons/s/nm/pixel.

    One entry per source term and wavelength node of ``sampling.quadrature``.
    Call integrate() on the returned record to integrate its spectral nodes.
    No QE or survival is applied. Local zodi comes from ``scene.local_zodi`` (None
    disables it); the speckle field from the selected instrument (None adds
    none). Thermal is the warm telescope's uniform emission from the selected
    instrument's ``warm_emission`` (zero when None); companion terms are zero.

    The scene is read at ``exposure.on_epoch(scene.epoch).time_d`` and the
    speckle field at the event stated on the field's own epoch.

    The coronagraph renders only what its data hold. ``supported`` of shape
    (W, ny, nx) is False at a detector pixel whose footprint, with the reach of
    each product's reconstruction, leaves any product the render reads at any
    node of the band, the speckle field included; the planet, star leakage,
    local zodi and exozodi are zero there at every node. ``static_supported``
    is the same flag without the speckle field. A planet outside the
    coronagraph's source domain contributes zero and leaves every other planet
    as it would be alone.

    Args:
        scene: The scene to render.
        observatory: The observatory to render it with.
        exposure: The commanded event giving the time, roll and instrument.
        sampling: The wavelength rule built for this event.
        static_beyond_speckle: Keep the static light where only the speckle
            field is missing. By default the product terms are zero wherever
            ``supported`` is False. With True they are zero only where
            ``static_supported`` is False: where that flag is True and
            ``supported`` is False the record is the static-only model, with
            no drift beyond the speckle field, and its star leakage is the
            static leakage plus the field's stored samples, zero-filled, under
            the field's own reconstruction. Both flags are the same as in the
            default render. A frame drawn from such a record (``detector.draw``
            on it) carries no flags, so an injection and recovery made that way
            has no drift to recover beyond the field, and its frame peak, which
            sets the tolerance of the detector's negative-rate floor, can be
            higher than the default's. :func:`observe` has no such option.

    Returns:
        The photon integrands in photons/s/nm/pixel, with the ``supported``
        and ``static_supported`` flags of shape (W, ny, nx).

    Raises:
        NotImplementedError: The exposure has an explicit pointing offset.
        ValueError: The event bands differ from the sampling bands, or the
            instrument is not named in the observatory.
        StampMismatchError: The sampling was built for other hardware.
        PrecisionError: A float32 event time would lose an epoch difference.
    """
    sel, coro, *clocks = _host_stage(scene, observatory, exposure, sampling)
    return _run(
        scene,
        sel,
        coro,
        exposure,
        clocks,
        sampling,
        "photons",
        static_beyond_speckle=static_beyond_speckle,
    )


def expected(
    scene: Scene,
    observatory: Observatory,
    exposure: Exposure,
    *,
    sampling: SpectralSampling,
    static_beyond_speckle: bool = False,
) -> Rates:
    """Stream nodes into (W,ny,nx) electrons/s with detector-owned yield.

    These are the expected rates for the instantaneous scene at the event time;
    ``exposure.plan`` does not enter. The exposure and the sampling may be
    arguments of JIT, vmap and grad: node wavelengths, weights, time and roll
    are traced, and one program serves any of their values. Each scan step
    sends one photon image and its wavelength and weight to the detector.
    Forward storage includes the band accumulator, one node's images and any
    shared planet PSFs for the band. JAX reverse-mode may retain residuals.

    ``supported`` of shape (W, ny, nx) is False where any node of the band
    reads beyond a product's data (see :func:`photon_rates`); the band-integrated
    planet, star leakage, local zodi and exozodi are zero there.
    ``static_supported`` is the same flag without the speckle field.

    Args:
        scene: The scene to render.
        observatory: The observatory to render it with, including its detector.
        exposure: The commanded event giving the time, roll and instrument.
        sampling: The wavelength rule built for this event.
        static_beyond_speckle: Keep the static light where only the speckle
            field is missing. By default the product terms are zero wherever
            ``supported`` is False. With True they are zero only where
            ``static_supported`` is False: where that flag is True and
            ``supported`` is False the record is the static-only model, with
            no drift beyond the speckle field, and its star leakage is the
            static leakage plus the field's stored samples, zero-filled, under
            the field's own reconstruction. Both flags are the same as in the
            default render. A frame drawn from such a record (``detector.draw``
            on it) carries no flags, so an injection and recovery made that way
            has no drift to recover beyond the field, and its frame peak, which
            sets the tolerance of the detector's negative-rate floor, can be
            higher than the default's. :func:`observe` has no such option.

    Returns:
        The expected rates in electrons/s, band integrated, shape (W, ny, nx),
        with the ``supported`` and ``static_supported`` flags of shape
        (W, ny, nx).

    Raises:
        NotImplementedError: The exposure has an explicit pointing offset.
        ValueError: The event bands differ from the sampling bands, or the
            instrument is not named in the observatory.
        StampMismatchError: The sampling was built for other hardware.
        PrecisionError: A float32 event time would lose an epoch difference.
    """
    sel, coro, *clocks = _host_stage(scene, observatory, exposure, sampling)
    return _run(
        scene,
        sel,
        coro,
        exposure,
        clocks,
        sampling,
        "electrons",
        static_beyond_speckle=static_beyond_speckle,
    )


def observe(
    key,
    scene: Scene,
    observatory: Observatory,
    exposure: Exposure,
    *,
    sampling: SpectralSampling,
) -> ExposureData:
    """Draw one detector observation and retain its exposure.

    Rates describe the instantaneous scene at the event time, held throughout
    ``exposure.plan``, which is required. Plan fields are scalar or (W,), with W
    matching the band axis. Detector noise is drawn once after combining all
    source terms. The roll lives on ``data.exposure.pointing``, not on the
    image grid. The variance and flags are absent and every pixel is marked
    valid. ``built_at`` records the producer, its version, the detector class
    and the sampling's ``layout`` stamp, so two observations built from one
    sampling stack.

    Args:
        key: JAX PRNG key for the detector noise draw.
        scene: The scene to render.
        observatory: The observatory to render with, including its detector.
        exposure: The commanded event, with the required frame plan.
        sampling: The wavelength rule built for this event.

    Returns:
        The measured electrons, with axes (band, y, x), and their exposure.

    Raises:
        NotImplementedError: The exposure has an explicit pointing offset.
        ValueError: The exposure has no frame plan, the event bands differ from
            the sampling bands, or the instrument is not named in the
            observatory.
        StampMismatchError: The sampling was built for other hardware.
    """
    _require_on_axis(exposure)
    if exposure.plan is None:
        raise ValueError("observe needs an exposure built with a FramePlan")
    sel = observatory.select(exposure.instrument)
    rates = expected(scene, observatory, exposure, sampling=sampling)
    electrons = _draw(sel.detector, key, rates, exposure.plan)
    image = Image(
        electrons,
        _detector_grid(sel, None),
        ("band",),
        Measure.integral(),
        "electrons",
    )
    layout = dict(sampling.built_at).get("layout", stamp_value(None))
    built_at = merge_stamps(
        build_stamps(
            producer="coronagraphoto.observe",
            producer_version=__version__,
            detector=type(sel.detector).__name__,
        ),
        (("layout", layout),),
    )
    return ExposureData(image, exposure, built_at=built_at)
