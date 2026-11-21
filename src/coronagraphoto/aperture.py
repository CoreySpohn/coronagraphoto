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
from optixstuff.spectral import Quadrature
from orbix.system import SkyFrame
from skyscapes import AbstractLocalZodi, Scene

from coronagraphoto.simulation import _run

# Successive rings start a golden-ratio fraction of the angular step apart, so
# the rings together sample every pixel phase and no azimuth is favoured.
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0

# The fields of a read that vary; thermal and companion terms are zero.
_READ_FIELDS = ("planet", "star_leakage", "local_zodi", "exozodi")


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
            each the mean over a ring of sources at that separation. NaN where
            the coronagraph has no response model.
        throughput_ring_std: Standard deviation of the first moment around
            each ring, shape (W, Q, K): how much azimuth and pixel phase the
            ring mean hides.
        kernels: Aperture weights on detector pixels for a centre at each
            half-pixel phase, shape (2, W, 2, 2, n, n): moment, band, y
            phase, x phase.
        probe: Aperture fraction of one source per band, shape (W,): the
            middle node of the band, a source on the +x axis at the middle
            separation that has a response. :func:`aperture_record` repeats
            the measurement to refuse a response built for another
            instrument. NaN for a band with no response anywhere.
    """

    rule: Circular = eqx.field(static=True)
    quadrature: Quadrature
    radius_pix: Array
    separation_step_arcsec: Array
    throughput: Array
    throughput_ring_std: Array
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
            detector pixel position (i + a / 2, j + b / 2). NaN where the
            aperture leaves the detector.
        uniform_sky: The same sums for a uniform sky of 1 photon/s/m^2/nm/sr,
            the local zodi per unit radiance; zero when the record has no
            local zodi.
        exozodi: The same sums for the scene's disk.
        n_pix: Sum of the aperture weights at each lattice position, shape
            (2, W, 2 ny, 2 nx).
        local_zodi: The scene's local-zodi model, or None for no local zodi.
        sky_frame: The scene's ``SkyFrame``: its right ascension and
            declination (None for an unplaced scene) and the epoch, which is
            day zero of the read's ``time_d``.
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
    local_zodi: AbstractLocalZodi | None
    sky_frame: SkyFrame
    reads: tuple[str, ...] = eqx.field(static=True)


def _weights(rule, frame, shape, radius_pix, dx_pix, dy_pix):
    """Coverage of each pixel by the rule's circle, offset from the grid centre."""
    step = rule.radius_lod / radius_pix
    grid = PixelGrid.regular(shape, step, "lod", frame=frame)
    ones = Image(jnp.ones(shape), grid, (), Measure.integral(), "stellar_flux_fraction")
    return np.asarray(rule.weights(ones, (dx_pix * step, dy_pix * step)))


@eqx.filter_jit
def _ring_moments(model, detector, wavelengths_nm, x_arcsec, y_arcsec, weights):
    """Aperture fractions (Q, 2, A) of A sources at each node of one band."""
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
        return jnp.where(psfs.valid, moments, jnp.nan)

    return jax.lax.map(node, wavelengths_nm)


def _band_models(instrument, quadrature, rule):
    """Per band: coronagraph model, lambda/D frame and aperture radius in pixels."""
    detector, coro = instrument.detector, instrument.coronagraph
    scale = float(detector.pixel_scale_arcsec)
    models, frames, radii = [], [], []
    for band in quadrature.bands:
        model = (
            coro.band(band.center_nm)
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


def _probe_index(throughput_row):
    """Index of the middle separation that has a response, or None."""
    finite = np.flatnonzero(np.isfinite(np.asarray(throughput_row)))
    return None if finite.size == 0 else int(finite[finite.size // 2])


def _probe(model, instrument, rule, frame, radius, nodes_nm, throughput, step):
    """Aperture fraction of one source on the +x axis at the middle node of a band.

    ``throughput`` is the band's ``(2, Q, K)`` table and ``step`` its separation
    spacing; both fix the source, so the build and the check measure the same one.
    """
    k = _probe_index(throughput[0, len(nodes_nm) // 2])
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
    moments = _ring_moments(
        model,
        detector,
        nodes_nm[len(nodes_nm) // 2 :][:1],
        np.array([separation]),
        np.array([0.0]),
        weights[None],
    )
    return float(moments[0, 0, 0])


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
            np.asarray(response.throughput[:, w]),
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
    per wavelength node. Sources sit on ``n_separations`` rings, evenly spaced
    from zero to ``max_separation_arcsec`` (by default the largest separation
    whose aperture stays on the detector), ``n_angles`` to a ring. Each source
    is imaged on the detector pixels by ``coronagraph.source_psfs`` and summed
    in an aperture centred on it.

    ``quadrature`` must be the one the exposures will carry, for example
    ``coronagraphoto.build_exposure(...).quadrature``. Only a
    ``Circular(radius_lod, centre="source")`` rule can be tabulated.
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
    throughput = np.empty((2, *nodes.shape, n_separations))
    ring_std = np.empty((*nodes.shape, n_separations))
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
            moments = np.asarray(
                _ring_moments(model, detector, nodes[w], x, y, weights)
            )
            throughput[:, w, :, k] = moments.mean(axis=-1).T
            ring_std[w, :, k] = moments[:, 0].std(axis=-1)
    probe = [
        _probe(
            model,
            instrument,
            rule,
            frame,
            radius,
            nodes[w],
            throughput[:, w],
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
        jnp.asarray(kernels),
        jnp.asarray(probe),
    )


def _lattice(maps, kernels, radius_pix):
    """Sum per-pixel maps (W, B, ny, nx) in apertures, to (2, W, B, 2 ny, 2 nx)."""
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
        # An aperture cut by the detector edge has no complete sum.
        inside = [
            (jnp.arange(2 * n) / 2 - radius_pix[w] >= -0.5)
            & (jnp.arange(2 * n) / 2 + radius_pix[w] <= n - 0.5)
            for n in (ny, nx)
        ]
        bands.append(jnp.where(inside[0][:, None] & inside[1], sums, jnp.nan))
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
    then summed in the response's aperture at every half-pixel position.
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

    The record's ``reads`` lists the scene paths it was built from: the ``star``
    and the ``dynamics.frame`` (distance, position and epoch), plus the
    ``local_zodi`` when there is one and the ``disk`` with its two midplane angles
    when there is one. The planets and their orbits are not read.
    """
    quadrature = exposure.quadrature
    nodes = quadrature.nodes_nm
    _check_response(instrument, quadrature, response)
    components = _run(
        scene, instrument, exposure, "photons", planets=False, uniform_sky=True
    )
    maps = jnp.stack(
        [components[name] for name in ("star_leakage", "local_zodi", "exozodi")],
        axis=1,
    )
    n_bands, n_nodes = nodes.shape
    shape = maps.shape[-2:]
    background = _lattice(
        maps.reshape(n_bands, 3 * n_nodes, *shape),
        response.kernels,
        response.radius_pix,
    ).reshape(2, n_bands, 3, n_nodes, 2 * shape[0], 2 * shape[1])
    star_leakage, uniform_sky, exozodi = jnp.moveaxis(background, 2, 0)
    n_pix = _lattice(
        jnp.ones((n_bands, 1, *shape)), response.kernels, response.radius_pix
    )[:, :, 0]

    def per_contrast(wavelength_nm):
        gain = instrument.effective_area_m2(wavelength_nm)
        return gain * scene.star.photon_flux_density(
            wavelength_nm, jnp.asarray(exposure.time_d), dist_pc=scene.dist_pc
        )

    star = jax.vmap(per_contrast)(nodes.ravel()).reshape(nodes.shape)
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
        scene.local_zodi,
        scene.sky_frame,
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
    """Read the last axis of a table at a fractional index; NaN outside it."""
    last = table.shape[-1] - 1
    k = jnp.clip(jnp.floor(position), 0, last - 1).astype(int)
    pair = jax.lax.dynamic_slice_in_dim(table, k, 2, axis=-1)
    value = pair[..., 0] + (position - k) * (pair[..., 1] - pair[..., 0])
    return jnp.where((position >= 0) & (position <= last), value, jnp.nan)


def _bilinear(table, row, column):
    """Read the last two axes of a table at a fractional index; NaN outside."""
    lead = table.shape[:-2]
    last_row, last_column = table.shape[-2] - 1, table.shape[-1] - 1
    i = jnp.clip(jnp.floor(row), 0, last_row - 1).astype(int)
    j = jnp.clip(jnp.floor(column), 0, last_column - 1).astype(int)
    # One slice of the four neighbours, so a read never touches a full map.
    corners = jax.lax.dynamic_slice(table, (*(0,) * len(lead), i, j), (*lead, 2, 2))
    value = _linear(_linear(corners, column - j), row - i)
    inside = (row >= 0) & (row <= last_row) & (column >= 0) & (column <= last_column)
    return jnp.where(inside, value, jnp.nan)


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
    The aperture is centred on the planet.

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
    positions. Everything is NaN where the aperture leaves the detector or the
    separation leaves the planet table. Thermal and companion terms are zero.

    All arguments are scalars apart from ``contrast``. Batch with
    ``jax.vmap``; a ``time_d`` shared by a batch of planets (``in_axes`` None,
    or closed over) costs one radiance evaluation for the whole batch.
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
    star_leakage = _bilinear(record.star_leakage, *here)
    local_zodi = _bilinear(record.uniform_sky, *here)
    exozodi = _bilinear(record.exozodi, *fixed)
    if record.local_zodi is not None:
        local_zodi = local_zodi * record.local_zodi.photon_radiance(
            record.quadrature.nodes_nm, record.sky_frame, time_d
        )
    planet = contrast * _linear(
        record.planet_per_contrast, separation / record.separation_step_arcsec
    )
    n_pix = _bilinear(record.n_pix, *here)

    def rates(moment, variance=None):
        return Rates(
            record.quadrature,
            "photon",
            planet=planet[moment],
            star_leakage=star_leakage[moment],
            local_zodi=local_zodi[moment],
            exozodi=exozodi[moment],
            n_pix=n_pix[moment],
            variance=variance,
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
    contrast from ``scene.contrasts`` at every wavelength node of the record.
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
    A scene with a batch of systems is not read; map this function over them.
    """
    time_d = jnp.asarray(time_d)
    if time_d.ndim != 1:
        raise ValueError("time_d must have shape (T,)")
    if scene.batch_shape:
        raise ValueError("scene_aperture_rates reads one system; vmap it over a batch")
    east, north = scene.positions(time_d)
    nodes = record.quadrature.nodes_nm
    contrast = jax.vmap(
        jax.vmap(lambda wavelength: scene.contrasts(wavelength, time_d))
    )(nodes)

    def read(east_arcsec, north_arcsec, contrast, time):
        rates = aperture_rates(
            record, east_arcsec, north_arcsec, contrast, time_d=time, roll_deg=roll_deg
        )
        return tuple(
            (
                *(getattr(r, name) for name in _READ_FIELDS),
                r.n_pix,
            )
            for r in (rates, rates.variance)
        )

    over_times = jax.vmap(read, in_axes=(0, 0, -1, 0), out_axes=-1)
    mean, variance = jax.vmap(over_times, in_axes=(0, 0, -2, None), out_axes=-2)(
        east, north, contrast, time_d
    )

    def rates(values, variance=None):
        *fields, n_pix = (jnp.where(scene.valid[:, None], v, 0.0) for v in values)
        return Rates(
            record.quadrature,
            "photon",
            **dict(zip(_READ_FIELDS, fields, strict=True)),
            n_pix=n_pix,
            variance=variance,
        )

    return rates(mean, rates(variance))
