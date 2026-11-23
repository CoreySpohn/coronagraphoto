"""Photometric-aperture rates read from tables built on the image path.

Three steps, each paid at its own cadence:

- :func:`aperture_response`, once per instrument and band set, tabulates
  the planet's aperture fraction against separation and the aperture's pixel
  weights. It holds nothing about a target.
- :func:`aperture_record`, once per target and exposure, renders the
  background images of :func:`coronagraphoto.photon_rates` and sums them in
  the aperture at every position of a half-pixel lattice.
- :func:`aperture_rates`, once per planet and observation time, interpolates
  both tables and scales the local zodi to the sightline of that time. It is
  a few gathers, so it can sit under ``jax.vmap`` over millions of planets.
  :func:`scene_aperture_rates` maps it over a scene's planets and times.

Every number comes from the same coronagraph products as the images, so an
aperture sum of :func:`coronagraphoto.expected` images and a read of the
record agree to the interpolation error of the tables.

The aperture is a circle fixed in angle within a band: its radius is
``rule.radius_lod`` at the band centre for every node of that band. Sums use
exact fractional pixel coverage ``w``; the second moment of each table uses
``w**2`` and carries the Poisson variance of the weighted sum.

A read holds only what the coronagraph data hold. It is supported when the
aperture lies on the detector, every pixel under it is supported on the image
path (:func:`coronagraphoto.expected`), and the separation lies in the planet
table at positions the coronagraph tabulates. An unsupported read is zero in
every field with ``Rates.supported`` False, never NaN, so a sum or a gradient
through it stays finite.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array
from optixstuff import Exposure, Instrument, Rates
from optixstuff.apertures import DEFAULT_APERTURE_RULE, Circular
from optixstuff.coronagraph import AbstractImagingCoronagraph, MultiBandCoronagraph
from optixstuff.grids import Image, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Quadrature, Resolution
from orbix.system import SkyFrame
from skyscapes import AbstractLocalZodi, Scene

from coronagraphoto.simulation import _real_counts, _run

# Successive rings start a golden-ratio fraction of the angular step apart, so
# the rings together sample every pixel phase and no azimuth is favoured.
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0

# The fields of a read that vary; companion terms are zero.
_READ_FIELDS = ("planet", "star_leakage", "local_zodi", "exozodi", "thermal")


class ApertureResponse(eqx.Module):
    """Target-independent aperture tables of one instrument and band set.

    Attributes:
        rule: The aperture rule the tables were summed with.
        quadrature: The wavelength nodes the tables are sampled at.
        radius_pix: Aperture radius in detector pixels, shape (W,).
        separation_step_arcsec: Spacing of the separation axis, which starts
            at zero.
        throughput: Fraction of a point source's light summed in an aperture
            centred on it, shape (2, W, Q, K): weights ``w`` then ``w**2``,
            each the mean over a ring of sources at that separation. Zero
            where the ring is not supported.
        throughput_ring_std: Standard deviation of the first moment around
            each ring, shape (W, Q, K): how much azimuth and pixel phase the
            ring mean hides.
        throughput_supported: Whether each ring lies in the coronagraph's
            source domain at that node, shape (W, Q, K).
        kernels: Aperture weights on detector pixels for a centre at each
            half-pixel phase, shape (2, W, 2, 2, n, n): moment, band, y
            phase, x phase.
        probe: Aperture fraction of one source per band, shape (W,): the
            middle node of the band, a source on the +x axis at the middle
            separation that has a response. :func:`aperture_record` repeats
            the measurement to refuse a response built for another
            instrument. NaN for a band with no supported ring anywhere (a
            build check, not a rate).
    """

    rule: Circular = eqx.field(static=True)
    quadrature: Quadrature
    radius_pix: Array
    separation_step_arcsec: Array
    throughput: Array
    throughput_ring_std: Array
    throughput_supported: Array
    kernels: Array
    probe: Array


class ApertureRecord(eqx.Module):
    """Aperture rates of one target and exposure, tabulated for reading.

    Rates are photons/s/nm at the nodes of ``quadrature``; the detector's
    yield is applied by the consumer, so one record serves any detector noise
    setting. Read it with :func:`aperture_rates`.

    The local zodi is the instrument's response to a uniform sky times a
    radiance that depends on the sightline, which moves with the Sun. The
    record holds the response and what the radiance is computed from, so one
    record serves any observation time.

    Attributes:
        quadrature: The exposure's wavelength nodes and weights.
        roll_deg: Roll the backgrounds were rendered at.
        pixel_scale_arcsec: Detector pixel pitch; the lattice pitch is half.
        separation_step_arcsec: Spacing of the planet table's separation axis.
        planet_per_contrast: Planet rate per unit planet-to-star contrast,
            shape (2, W, Q, K).
        star_leakage: Star leakage summed in an aperture centred on each
            lattice position, shape (2, W, Q, 2 ny, 2 nx): moment, band,
            node, then the lattice. Lattice index (2 i + a, 2 j + b) is
            detector pixel position (i + a / 2, j + b / 2). Zero where the
            aperture leaves the detector.
        uniform_sky: The same sums for a uniform sky of 1 photon/s/m^2/nm/sr,
            the local zodi per unit radiance; zero when the record has no
            local zodi.
        exozodi: The same sums for the scene's disk.
        n_pix: Sum of the aperture weights at each lattice position, shape
            (2, W, 2 ny, 2 nx); zero where the aperture leaves the detector.
        unsupported: One where the aperture at a lattice position leaves the
            detector or covers a pixel :func:`coronagraphoto.expected` does not
            support at some node of the band, else zero, shape (W, 2 ny, 2 nx).
            A float, so a bilinear read of it is positive exactly when a
            neighbour the read weights is unsupported.
        planet_unsupported: One where the planet table's ring is not in the
            coronagraph's source domain, else zero, shape (W, Q, K).
        thermal: The warm telescope's emission per detector pixel at each
            node, shape (W, Q); zero without ``warm_emission``. The emission
            is uniform, so a read is this times the aperture's ``n_pix``.
        local_zodi: The scene's local-zodi model, or None for no local zodi.
        sky_frame: The scene's ``SkyFrame``: its right ascension and
            declination (None for an unplaced scene) and the epoch, which is
            day zero of the read's ``time_d``.
        resolution: The exposure's ``Resolution``, which reduces a tabulated
            local zodi (and, in :func:`scene_aperture_rates`, the scene's
            sources) on ``quadrature``.
        reads: Paths, relative to the scene, of every scene value the record was
            built from, each a whole leaf or a node covering the paths under it.
            Changing one of them leaves the record stale; rebuild it or hold the
            value.
    """

    quadrature: Quadrature
    roll_deg: Array
    pixel_scale_arcsec: Array
    separation_step_arcsec: Array
    planet_per_contrast: Array
    star_leakage: Array
    uniform_sky: Array
    exozodi: Array
    n_pix: Array
    unsupported: Array
    planet_unsupported: Array
    thermal: Array
    local_zodi: AbstractLocalZodi | None
    sky_frame: SkyFrame
    resolution: Resolution
    reads: tuple[str, ...] = eqx.field(static=True)


def _weights(rule, frame, shape, radius_pix, dx_pix, dy_pix):
    """Coverage of each pixel by the rule's circle, offset from the grid centre."""
    step = rule.radius_lod / radius_pix
    grid = PixelGrid.regular(shape, step, "lod", frame=frame)
    ones = Image(jnp.ones(shape), grid, (), Measure.integral(), "stellar_flux_fraction")
    return np.asarray(rule.weights(ones, (dx_pix * step, dy_pix * step)))


@eqx.filter_jit
def _ring_moments(model, detector, wavelengths_nm, x_arcsec, y_arcsec, weights):
    """Aperture fractions (Q, 2, A) of A sources at each node of one band.

    Also returns the (Q, A) flags of sources inside the source domain. A source
    outside it has a zero image and so a zero fraction.
    """
    grid = detector.grid

    def node(wavelength_nm):
        native = model.at_wavelength(wavelength_nm).source_grid
        scale = native.frame.lod_to_arcsec(wavelength_nm)
        pixels = replace(
            grid.to_lod(wavelength_nm, native.frame), orientation=native.orientation
        )
        psfs = model.source_psfs(
            wavelength_nm, x_arcsec / scale, y_arcsec / scale, grid=pixels
        )
        moments = jnp.stack(
            [
                jnp.einsum("ayx,ayx->a", weights**power, psfs.image.data)
                for power in (1, 2)
            ]
        )
        return jnp.where(psfs.valid, moments, 0.0), psfs.valid

    return jax.lax.map(node, wavelengths_nm)


def _band_models(instrument, quadrature, rule):
    """Per band: coronagraph model, lambda/D frame and aperture radius in pixels."""
    detector, coro = instrument.detector, instrument.coronagraph
    scale = float(detector.pixel_scale_arcsec)
    models, frames, radii = [], [], []
    for band in quadrature.bands:
        model = (
            coro.coronagraph_at(band.center_nm)
            if isinstance(coro, MultiBandCoronagraph)
            else coro
        )
        if not isinstance(model, AbstractImagingCoronagraph):
            raise TypeError("aperture tables require an AbstractImagingCoronagraph")
        frame = model.at_wavelength(band.center_nm).source_grid.frame
        models.append(model)
        frames.append(frame)
        radii.append(
            rule.radius_lod * float(frame.lod_to_arcsec(band.center_nm)) / scale
        )
    return models, frames, radii


def _probe_index(supported_row):
    """Index of the middle supported separation, or None."""
    inside = np.flatnonzero(np.asarray(supported_row))
    return None if inside.size == 0 else int(inside[inside.size // 2])


def _probe(model, instrument, rule, frame, radius, nodes_nm, supported, step):
    """Aperture fraction of one source on the +x axis at the middle node of a band.

    ``supported`` is the band's ``(Q, K)`` ring table and ``step`` its separation
    spacing; both fix the source, so the build and the check measure the same one.
    """
    k = _probe_index(supported[len(nodes_nm) // 2])
    if k is None:
        return np.nan
    detector = instrument.detector
    separation = k * float(step)
    weights = _weights(
        rule,
        frame,
        detector.grid.shape,
        radius,
        separation / float(detector.pixel_scale_arcsec),
        0.0,
    )
    moments, _ = _ring_moments(
        model,
        detector,
        nodes_nm[len(nodes_nm) // 2 :][:1],
        np.array([separation]),
        np.array([0.0]),
        weights[None],
    )
    return float(moments[0, 0, 0])


def _padded_slots(quadrature):
    """Per band, the real node each of the (Q,) slots repeats, as host arrays.

    A real slot is its own node; a padded slot repeats the first real node of
    its band, as the image routes do.
    """
    q = quadrature.nodes_nm.shape[1]
    return [
        np.where(np.arange(q) < real, np.arange(q), 0)
        for real in _real_counts(quadrature)
    ]


def _concrete(*trees):
    """True when no array leaf of the trees is traced."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(trees, eqx.is_array))
    return not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _check_response(instrument, quadrature, response):
    """Raise a ValueError when ``response`` was built for something else.

    Compared in order of cost: the wavelength nodes and bands, the aperture
    radius in pixels (which carries each band's lambda/D frame, the rule and
    the detector pixel scale), then one measured source per band, which
    carries the coronagraph's own tables. Traced inputs are not checked. The
    comparison runs eagerly even inside ``jax.jit``, so a closed-over
    instrument is checked once at trace time.
    """
    if not _concrete(instrument, quadrature, response):
        return
    with jax.ensure_compile_time_eval():
        _compare_response(instrument, quadrature, response)


def _compare_response(instrument, quadrature, response):
    nodes, built = (
        np.asarray(quadrature.nodes_nm),
        np.asarray(response.quadrature.nodes_nm),
    )
    if nodes.shape != built.shape or not np.array_equal(nodes, built):
        raise ValueError("the response was built for other wavelength nodes")
    if quadrature.bands != response.quadrature.bands:
        raise ValueError("the response was built for other bands")
    models, frames, radii = _band_models(instrument, quadrature, response.rule)
    if not np.allclose(radii, np.asarray(response.radius_pix), rtol=1e-6):
        raise ValueError(
            "the response was built for another detector pixel scale or "
            "lambda/D frame: its aperture radius is "
            f"{np.asarray(response.radius_pix).tolist()} pixels, this instrument's "
            f"is {radii}"
        )
    for w, (model, frame) in enumerate(zip(models, frames, strict=True)):
        expected = float(response.probe[w])
        measured = _probe(
            model,
            instrument,
            response.rule,
            frame,
            float(response.radius_pix[w]),
            nodes[w],
            np.asarray(response.throughput_supported[w]),
            response.separation_step_arcsec,
        )
        if not np.isclose(measured, expected, rtol=1e-4, atol=1e-7, equal_nan=True):
            raise ValueError(
                f"the response was built for another coronagraph or detector: "
                f"a source in band {w} puts {measured:.6g} of its light in the "
                f"aperture, the response recorded {expected:.6g}"
            )


def aperture_response(
    instrument: Instrument,
    quadrature: Quadrature,
    *,
    rule: Circular = DEFAULT_APERTURE_RULE,
    n_separations: int = 64,
    n_angles: int = 16,
    max_separation_arcsec: float | None = None,
) -> ApertureResponse:
    """Tabulate an instrument's aperture throughput and pixel weights.

    A host-side build, run once per instrument and band set and reused for
    every target: it costs ``n_separations * n_angles`` point-source images
    per real wavelength node. A padded slot of a band with fewer nodes than
    the widest is not imaged; its tables repeat the band's first real node.
    Sources sit on ``n_separations`` rings, evenly spaced from zero to
    ``max_separation_arcsec`` (by default the largest separation whose
    aperture stays on the detector), ``n_angles`` to a ring. Each source
    is imaged on the detector pixels by ``coronagraph.source_psfs`` and summed
    in an aperture centred on it.

    ``quadrature`` must be the one the exposures will carry, for example
    ``coronagraphoto.build_exposure(...).quadrature``. Only a
    ``Circular(radius_lod, centre="source")`` rule can be tabulated.

    Args:
        instrument: The instrument whose detector and coronagraph the tables
            are built for.
        quadrature: The wavelength nodes and bands the tables are sampled at.
        rule: The aperture rule to sum with. Must be a ``Circular`` rule with
            ``centre="source"``.
        n_separations: Number of rings on the separation axis, at least two.
        n_angles: Number of sources on each ring, at least one.
        max_separation_arcsec: Outermost ring separation in arcsec. None uses
            the largest separation whose aperture stays on the detector.

    Returns:
        The tables of throughput, ring statistics and pixel weights for this
        instrument and band set.
    """
    if not isinstance(rule, Circular) or rule.centre != "source":
        raise NotImplementedError(
            'aperture tables need a Circular rule with centre="source"'
        )
    if n_separations < 2 or n_angles < 1:
        raise ValueError("need at least two separations and one angle")
    detector = instrument.detector
    grid = detector.grid
    scale = float(detector.pixel_scale_arcsec)
    models, frames, radii = _band_models(instrument, quadrature, rule)
    if max_separation_arcsec is None:
        max_separation_arcsec = (min(grid.shape) / 2 - 1 - max(radii)) * scale
    if max_separation_arcsec <= 0:
        raise ValueError("the aperture does not fit on the detector")
    separations = np.linspace(0.0, max_separation_arcsec, n_separations)

    nodes = quadrature.nodes_nm
    slots = _padded_slots(quadrature)
    counts = _real_counts(quadrature)
    throughput = np.empty((2, *nodes.shape, n_separations))
    ring_std = np.empty((*nodes.shape, n_separations))
    ring_supported = np.empty((*nodes.shape, n_separations), bool)
    size = 2 * int(np.ceil(max(radii) + 1)) + 1
    kernels = np.empty((2, len(models), 2, 2, size, size))
    for w, (model, frame, radius) in enumerate(zip(models, frames, radii, strict=True)):
        for a, b in np.ndindex(2, 2):
            kernel = _weights(rule, frame, (size, size), radius, b / 2, a / 2)
            kernels[:, w, a, b] = kernel, kernel**2
        for k, separation in enumerate(separations):
            angle = 2 * np.pi * (np.arange(n_angles) + (k * _GOLDEN) % 1) / n_angles
            x, y = separation * np.cos(angle), separation * np.sin(angle)
            weights = np.stack(
                [
                    _weights(rule, frame, grid.shape, radius, dx, dy)
                    for dx, dy in zip(x / scale, y / scale, strict=True)
                ]
            )
            # Only the real nodes are imaged; a padded slot repeats the first.
            moments, inside = _ring_moments(
                model, detector, nodes[w, : counts[w]], x, y, weights
            )
            moments = np.asarray(moments)[slots[w]]
            throughput[:, w, :, k] = moments.mean(axis=-1).T
            ring_std[w, :, k] = moments[:, 0].std(axis=-1)
            ring_supported[w, :, k] = np.asarray(inside)[slots[w]].all(axis=-1)
    probe = [
        _probe(
            model,
            instrument,
            rule,
            frame,
            radius,
            nodes[w],
            ring_supported[w],
            separations[1],
        )
        for w, (model, frame, radius) in enumerate(
            zip(models, frames, radii, strict=True)
        )
    ]
    return ApertureResponse(
        rule,
        quadrature,
        jnp.asarray(radii),
        jnp.asarray(separations[1]),
        jnp.asarray(throughput),
        jnp.asarray(ring_std),
        jnp.asarray(ring_supported),
        jnp.asarray(kernels),
        jnp.asarray(probe),
    )


def _on_detector(radius_pix, ny, nx):
    """Whether the aperture at each lattice position is whole on the detector.

    Shape (W, 2 ny, 2 nx): a band's aperture of ``radius_pix`` pixels at
    detector position (i / 2, j / 2) lies inside the pixel edges.
    """
    inside = [
        (jnp.arange(2 * n) / 2 - radius_pix[:, None] >= -0.5)
        & (jnp.arange(2 * n) / 2 + radius_pix[:, None] <= n - 0.5)
        for n in (ny, nx)
    ]
    return inside[0][:, :, None] & inside[1][:, None, :]


def _lattice(maps, kernels, radius_pix):
    """Sum per-pixel maps (W, B, ny, nx) in apertures, to (2, W, B, 2 ny, 2 nx).

    An aperture cut by the detector edge has no complete sum and reads zero.
    """
    n_bands, n_maps, ny, nx = maps.shape
    size = kernels.shape[-1]
    bands = []
    for w in range(n_bands):
        # Cross-correlation: output pixel (i, j) under kernel phase (a, b) is
        # the aperture centred at pixel position (i + a / 2, j + b / 2).
        sums = jax.lax.conv_general_dilated(
            maps[w][:, None],
            kernels[:, w].reshape(8, 1, size, size),
            window_strides=(1, 1),
            padding="SAME",
        ).reshape(n_maps, 2, 2, 2, ny, nx)
        sums = sums.transpose(1, 0, 4, 2, 5, 3).reshape(2, n_maps, 2 * ny, 2 * nx)
        inside = _on_detector(radius_pix[w : w + 1], ny, nx)[0]
        bands.append(jnp.where(inside, sums, 0.0))
    return jnp.stack(bands, axis=1)


def aperture_record(
    scene: Scene,
    instrument: Instrument,
    exposure: Exposure,
    response: ApertureResponse,
) -> ApertureRecord:
    """Tabulate one target's aperture rates for one exposure.

    The star leakage (with the instrument's speckle field, if any), a uniform
    sky of unit radiance and the scene's disk are rendered once by the image
    path at ``exposure.time_d`` and ``exposure.roll_deg``, without planets,
    then summed in the response's aperture at every half-pixel position. Only
    the real nodes of each band are rendered and summed; a padded slot repeats
    the band's first real node.
    The record keeps ``scene.local_zodi`` and the scene's sky frame (position
    and epoch), from which :func:`aperture_rates` computes the local-zodi
    radiance at the time of each read; nothing of the local zodi is fixed at
    ``exposure.time_d``. A sightline-dependent local zodi on a scene with no sky
    position still builds a record; it raises a ``ValueError`` at the read, not
    here. The planet table is the response's throughput times the star's photon
    flux density and the collecting area and optical throughput, so a planet
    enters the read only as an east offset, a north offset and a contrast.

    ``response`` must have been built for this instrument and for
    ``exposure.quadrature``. On concrete inputs the call refuses another one
    with a ``ValueError`` naming what differs: the wavelength nodes or bands,
    the detector pixel scale or a band's lambda/D frame (through the aperture
    radius), or the coronagraph tables (through one source per band,
    measured again against the value the response stored). A coronagraph or
    detector swapped with ``equinox.tree_at`` therefore cannot pair its star
    leakage with the old planet throughput. The check costs one source image
    per band, small beside the build.

    The warm telescope's emission is uniform and carries no coronagraph factor,
    so the record keeps its per-pixel rate at each node (zero when the
    instrument has no ``warm_emission``) and a read multiplies it by the
    aperture's pixel count.

    The record's ``reads`` lists the scene paths it was built from: the ``star``
    and the ``dynamics.frame`` (distance, position and epoch), plus the
    ``local_zodi`` when there is one and the ``disk`` with its two midplane angles
    when there is one. The planets and their orbits are not read.

    The scene's ``flux_scale`` leaves enter as follows. The star's and the
    disk's scales are part of the nodes the record reads, so they are fixed at
    the build: changing one leaves the record stale. The local zodi's scale is
    applied at each read, through the kept model. A planet's scale is not
    read here; it multiplies the contrast the caller passes, which
    :func:`scene_aperture_rates` takes from the scene.

    Args:
        scene: The target scene, with its star, disk and local zodi.
        instrument: The instrument the exposure is rendered with.
        exposure: The exposure whose quadrature, ``time_d`` and ``roll_deg``
            the star leakage, uniform sky and disk are rendered at.
        response: The aperture tables built for this instrument and for
            ``exposure.quadrature``.

    Returns:
        The tabulated aperture rates in photons/s/nm at the nodes of the
        exposure's quadrature, ready to read with :func:`aperture_rates`.
    """
    quadrature = exposure.quadrature
    nodes = quadrature.nodes_nm
    _check_response(instrument, quadrature, response)
    images = _run(
        scene, instrument, exposure, "photons", planets=False, uniform_sky=True
    )
    maps = jnp.stack(
        [getattr(images, name) for name in ("star_leakage", "local_zodi", "exozodi")],
        axis=1,
    )
    n_bands = nodes.shape[0]
    shape = maps.shape[-2:]
    # Sum the real nodes of each band only; a padded slot repeats the first.
    background = jnp.stack(
        [
            _lattice(
                maps[w : w + 1, :, : _real_counts(quadrature)[w]].reshape(
                    1, -1, *shape
                ),
                response.kernels[:, w : w + 1],
                response.radius_pix[w : w + 1],
            ).reshape(2, 3, -1, 2 * shape[0], 2 * shape[1])[:, :, slots]
            for w, slots in enumerate(_padded_slots(quadrature))
        ],
        axis=1,
    )
    star_leakage, uniform_sky, exozodi = jnp.moveaxis(background, 2, 0)
    n_pix = _lattice(
        jnp.ones((n_bands, 1, *shape)), response.kernels, response.radius_pix
    )[:, :, 0]
    # An aperture is unsupported when it leaves the detector or any pixel under
    # it is: the weighted count of unsupported pixels is positive.
    bad = _lattice(
        (~images.supported).astype(float)[:, None],
        response.kernels,
        response.radius_pix,
    )[0, :, 0]
    unsupported = ~_on_detector(response.radius_pix, *shape) | (bad > 0)

    gain = jax.vmap(instrument.effective_area_m2)(nodes.ravel()).reshape(nodes.shape)
    star = (
        gain
        * scene.star.node_flux_density(
            quadrature,
            exposure.resolution,
            jnp.atleast_1d(jnp.asarray(exposure.time_d)),
            dist_pc=scene.dist_pc,
        )[..., 0]
    )
    return ApertureRecord(
        quadrature,
        exposure.roll_deg,
        jnp.asarray(instrument.detector.pixel_scale_arcsec),
        response.separation_step_arcsec,
        star[None, :, :, None] * response.throughput,
        star_leakage,
        uniform_sky,
        exozodi,
        n_pix,
        unsupported.astype(float),
        (~response.throughput_supported).astype(float),
        images.thermal[:, :, 0, 0],
        scene.local_zodi,
        scene.sky_frame,
        exposure.resolution,
        reads=(
            "star",
            "dynamics.frame",
            *(() if scene.local_zodi is None else ("local_zodi",)),
            *(
                ()
                if scene.disk is None
                else ("disk", "midplane_inc_deg", "midplane_position_angle_deg")
            ),
        ),
    )


def _linear(table, position):
    """Read the last axis of a table at a fractional index.

    Returns the value and whether the position lies inside the table. The
    read clamps to the table, so the value is finite and its gradient exact
    everywhere; outside the table it is the edge's linear continuation and the
    caller replaces it.
    """
    last = table.shape[-1] - 1
    k = jnp.clip(jnp.floor(position), 0, last - 1).astype(int)
    pair = jax.lax.dynamic_slice_in_dim(table, k, 2, axis=-1)
    value = pair[..., 0] + (position - k) * (pair[..., 1] - pair[..., 0])
    return value, (position >= 0) & (position <= last)


def _bilinear(table, row, column):
    """Read the last two axes of a table at a fractional index.

    Returns the value and whether the position lies inside the table, as
    :func:`_linear` does.
    """
    lead = table.shape[:-2]
    last_row, last_column = table.shape[-2] - 1, table.shape[-1] - 1
    i = jnp.clip(jnp.floor(row), 0, last_row - 1).astype(int)
    j = jnp.clip(jnp.floor(column), 0, last_column - 1).astype(int)
    # One slice of the four neighbours, so a read never touches a full map.
    corners = jax.lax.dynamic_slice(table, (*(0,) * len(lead), i, j), (*lead, 2, 2))
    value, _ = _linear(corners, column - j)
    value, _ = _linear(value, row - i)
    inside = (row >= 0) & (row <= last_row) & (column >= 0) & (column <= last_column)
    return value, inside


def aperture_rates(
    record: ApertureRecord,
    east_arcsec: ArrayLike,
    north_arcsec: ArrayLike,
    contrast: ArrayLike,
    *,
    time_d: ArrayLike,
    roll_deg: ArrayLike | None = None,
) -> Rates:
    """Read one planet's aperture rates, in photons/s/nm, from a record.

    The planet sits ``east_arcsec`` east and ``north_arcsec`` north of the
    star, the order and convention of ``Scene.positions``, with planet-to-star
    ``contrast``: a scalar, or shape (W, Q) for one value per wavelength node.
    A per-node contrast is the star-weighted node contrast, ``Sources.contrast``
    of ``Scene.sources``: the number that returns the planet's node-effective
    flux when multiplied by the node-effective star the record holds. A scalar
    is exact for a smooth contrast. The aperture is centred on the planet.

    ``time_d`` is the observation time in days from the record's epoch. It
    sets the local zodi: the record's zodi model is evaluated for the record's
    sky frame at that time, and its radiance at each wavelength node scales
    the record's uniform-sky table.
    It has no default, because the time a record was built at is not the time
    of a later read and the zodi would silently be that of the wrong season.
    The star's flux, the disk and any speckle realization stay those of the
    exposure the record was built from.

    ``roll_deg`` is the position angle of detector +y for this read and
    defaults to the record's roll. Star leakage, local zodi and the pixel
    counts are fixed on the detector, so they are read where the planet falls
    at this roll. The exozodi is fixed on the sky and is read where the planet
    fell at the record's roll.

    The returned ``Rates`` has fields of shape (W, Q), ``n_pix`` of shape (W,)
    and squared-weight ``variance`` rates. Pass it to
    :func:`optixstuff.snr` or :func:`optixstuff.exposure_time`, or to
    ``detector.electrons``. Planet rates are linear in separation between the
    response's rings; backgrounds are bilinear between half-pixel
    positions. A read is supported when the aperture is on the detector, every
    pixel under it is supported on the image path and the separation lies in the
    planet table at positions the coronagraph tabulates, at every node of the
    band. Otherwise every field is zero and ``supported`` (shape (W,)) is False.
    Thermal is the per-pixel rate of the record times ``n_pix``; companion
    terms are zero.

    All arguments are scalars apart from ``contrast``. Batch with
    ``jax.vmap``; a ``time_d`` shared by a batch of planets (``in_axes`` None,
    or closed over) costs one radiance evaluation for the whole batch.

    Args:
        record: The aperture record built for the target and exposure.
        east_arcsec: Planet offset east of the star, in arcsec.
        north_arcsec: Planet offset north of the star, in arcsec.
        contrast: Planet-to-star contrast, a scalar or the star-weighted node
            contrast of shape (W, Q).
        time_d: Observation time in days from the record's epoch.
        roll_deg: Position angle of detector +y for this read, in degrees.
            None uses the record's roll.

    Returns:
        The planet, star leakage, local zodi, exozodi and thermal rates in
        photons/s/nm, with ``n_pix``, ``variance`` and the ``supported`` flag.
    """
    east, north = jnp.asarray(east_arcsec), jnp.asarray(north_arcsec)
    separation = jnp.hypot(east, north)
    ny, nx = record.n_pix.shape[-2] // 2, record.n_pix.shape[-1] // 2

    def lattice_index(roll):
        x, y = SkyOrientation(roll).sky_to_detector(east, north)
        return (
            (ny - 1) + 2 * y / record.pixel_scale_arcsec,
            (nx - 1) + 2 * x / record.pixel_scale_arcsec,
        )

    fixed = lattice_index(record.roll_deg)
    here = fixed if roll_deg is None else lattice_index(roll_deg)
    star_leakage, inside_here = _bilinear(record.star_leakage, *here)
    local_zodi, _ = _bilinear(record.uniform_sky, *here)
    exozodi, inside_fixed = _bilinear(record.exozodi, *fixed)
    n_pix, _ = _bilinear(record.n_pix, *here)
    bad_here, _ = _bilinear(record.unsupported, *here)
    bad_fixed, _ = _bilinear(record.unsupported, *fixed)
    if record.local_zodi is not None:
        local_zodi = local_zodi * record.local_zodi.node_radiance(
            record.quadrature, record.resolution, record.sky_frame, time_d
        )
    position = separation / record.separation_step_arcsec
    planet_per_contrast, in_table = _linear(record.planet_per_contrast, position)
    planet = contrast * planet_per_contrast
    bad_ring, _ = _linear(record.planet_unsupported, position)
    supported = (
        inside_here
        & inside_fixed
        & in_table
        & (bad_here == 0)
        & (bad_fixed == 0)
        & jnp.all(bad_ring == 0, axis=-1)
    )

    def zero(x):
        return jnp.where(
            supported.reshape(supported.shape + (1,) * (x.ndim - 1)), x, 0.0
        )

    def rates(moment, variance=None):
        return Rates(
            record.quadrature,
            "photon",
            planet=zero(planet[moment]),
            star_leakage=zero(star_leakage[moment]),
            local_zodi=zero(local_zodi[moment]),
            exozodi=zero(exozodi[moment]),
            thermal=zero(record.thermal * n_pix[moment][:, None]),
            n_pix=zero(n_pix[moment]),
            variance=variance,
            supported=supported,
        )

    return rates(0, rates(1))


def scene_aperture_rates(
    record: ApertureRecord,
    scene: Scene,
    time_d: ArrayLike,
    *,
    roll_deg: ArrayLike | None = None,
) -> Rates:
    """Read every planet of a scene at every time, in photons/s/nm, from a record.

    A ``jax.vmap`` of :func:`aperture_rates` over the scene's planets and the
    times: positions come from ``scene.positions(time_d)`` and each planet's
    contrast from ``Scene.sources`` on the record's quadrature and resolution
    (the star-weighted node contrast, ``Sources.contrast``).
    Nothing is rebuilt. The scene should be the one the record was built from
    (``record.reads`` lists what the record holds fixed).

    ``time_d`` has shape ``(T,)``, days from the scene's epoch; ``roll_deg`` is
    as in :func:`aperture_rates`. The returned ``Rates`` follows the layout of
    every ``Rates`` with a quadrature: the band and node axes lead and the
    batch axes trail, so fields have shape ``(W, Q, K, T)``, ``n_pix`` has
    shape ``(W, K, T)``, and ``variance`` has the squared-weight rates. Pass it
    to :func:`optixstuff.snr` or :func:`optixstuff.exposure_time`, which return
    one value per band, planet and time. A planet with ``scene.valid`` False
    reads zero in every field, so its exposure time is invalid, not a number.
    ``supported`` has shape ``(W, K, T)``: False where the read is outside what
    the coronagraph data hold (see :func:`aperture_rates`), with zero fields.
    A scene with a batch of systems is not read; map this function over them.

    Args:
        record: The aperture record built for the scene's target.
        scene: The scene whose planets are read.
        time_d: Observation times, shape (T,), in days from the scene's epoch.
        roll_deg: Position angle of detector +y, in degrees, as in
            :func:`aperture_rates`. None uses the record's roll.

    Returns:
        The rates of every planet at every time in photons/s/nm, with the
        shapes given above.
    """
    time_d = jnp.asarray(time_d)
    if time_d.ndim != 1:
        raise ValueError("time_d must have shape (T,)")
    if scene.batch_shape:
        raise ValueError("scene_aperture_rates reads one system; vmap it over a batch")
    east, north = scene.positions(time_d)
    contrast = scene.sources(record.quadrature, record.resolution, time_d).contrast

    def read(east_arcsec, north_arcsec, contrast, time):
        rates = aperture_rates(
            record, east_arcsec, north_arcsec, contrast, time_d=time, roll_deg=roll_deg
        )
        return tuple(
            (
                *(getattr(r, name) for name in _READ_FIELDS),
                r.n_pix,
                r.supported,
            )
            for r in (rates, rates.variance)
        )

    over_times = jax.vmap(read, in_axes=(0, 0, -1, 0), out_axes=-1)
    mean, variance = jax.vmap(over_times, in_axes=(0, 0, -2, None), out_axes=-2)(
        east, north, contrast, time_d
    )

    def rates(values, variance=None):
        *fields, n_pix, supported = values
        fields = [jnp.where(scene.valid[:, None], v, 0.0) for v in fields]
        return Rates(
            record.quadrature,
            "photon",
            **dict(zip(_READ_FIELDS, fields, strict=True)),
            n_pix=jnp.where(scene.valid[:, None], n_pix, 0.0),
            variance=variance,
            supported=supported,
        )

    return rates(mean, rates(variance))
