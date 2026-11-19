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

Every number comes from the same coronagraph products as the images, so an
aperture sum of :func:`coronagraphoto.expected` images and a read of the
record agree to the interpolation error of the tables.

The aperture is a circle fixed in angle within a band: its radius is
``rule.radius_lod`` at the band centre for every node of that band. Sums use
exact fractional pixel coverage ``w``; the second moment of each table uses
``w**2`` and carries the Poisson variance of the weighted sum.
"""

from dataclasses import replace
from typing import ClassVar

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
from orbix.observatory import ecliptic_pointing
from orbix.system import SkyFrame
from skyscapes import Scene
from zodi.jax import AbstractLocalZodi

from coronagraphoto.simulation import _require_position, _run

_BACKGROUNDS = ("star_leakage", "local_zodi", "exozodi")
# Successive rings start a golden-ratio fraction of the angular step apart, so
# the rings together sample every pixel phase and no azimuth is favoured.
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


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
    """

    rule: Circular = eqx.field(static=True)
    quadrature: Quadrature
    radius_pix: Array
    separation_step_arcsec: Array
    throughput: Array
    throughput_ring_std: Array
    kernels: Array


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
        planet: Planet rate per unit planet-to-star contrast, shape
            (2, W, Q, K).
        background: Star leakage, local zodi and exozodi summed in an aperture
            centred on each lattice position, shape (3, 2, W, Q, 2 ny, 2 nx):
            source, then moment. The local zodi is per unit sky radiance:
            the rate for a uniform sky of 1 photon/s/m^2/nm/sr.
            Lattice index (2 i + a, 2 j + b) is detector pixel position
            (i + a / 2, j + b / 2). NaN where the aperture leaves the
            detector.
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
    planet: Array
    background: Array
    n_pix: Array
    local_zodi: AbstractLocalZodi | None
    sky_frame: SkyFrame
    reads: tuple[str, ...] = eqx.field(static=True)


class _UnitSky(AbstractLocalZodi):
    """A sky of 1 photon/s/m^2/nm/sr at every wavelength and sightline."""

    uses_pointing: ClassVar[bool] = False

    def spectral_breaks_nm(self):
        """No breaks: the radiance is constant."""
        return np.empty(0)

    def photon_radiance(
        self, wavelength_nm, *, ecliptic_latitude_deg, helioecliptic_longitude_deg
    ):
        """One, in the shape of the wavelengths."""
        return jnp.ones_like(wavelength_nm)


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
    detector, coro = instrument.detector, instrument.coronagraph
    grid = detector.grid
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
    return ApertureResponse(
        rule,
        quadrature,
        jnp.asarray(radii),
        jnp.asarray(separations[1]),
        jnp.asarray(throughput),
        jnp.asarray(ring_std),
        jnp.asarray(kernels),
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
    ``exposure.time_d``. The planet table is the response's throughput
    times the star's photon flux density and the collecting area and optical
    throughput, so a planet enters the read only as an east offset, a north
    offset and a contrast.

    ``response`` must have been built for this instrument and for
    ``exposure.quadrature``.

    The record's ``reads`` lists the scene paths it was built from: the ``star``
    and the ``dynamics.frame`` (distance, position and epoch), plus the
    ``local_zodi`` when there is one and the ``disk`` with its two midplane angles
    when there is one. The planets and their orbits are not read.
    """
    quadrature = exposure.quadrature
    nodes, built = quadrature.nodes_nm, response.quadrature.nodes_nm
    if nodes.shape != built.shape or (
        not isinstance(nodes, jax.core.Tracer)
        and not isinstance(built, jax.core.Tracer)
        and not np.array_equal(np.asarray(nodes), np.asarray(built))
    ):
        raise ValueError("the response was built for other wavelength nodes")
    _require_position(scene)
    local_zodi = scene.local_zodi
    if local_zodi is not None:
        scene = eqx.tree_at(lambda s: s.local_zodi, scene, _UnitSky())
    sky_frame = scene.dynamics.frame
    components = _run(scene, instrument, exposure, "photons", planets=False)
    maps = jnp.stack([components[name] for name in _BACKGROUNDS], axis=1)
    n_bands, n_nodes = nodes.shape
    shape = maps.shape[-2:]
    background = _lattice(
        maps.reshape(n_bands, 3 * n_nodes, *shape),
        response.kernels,
        response.radius_pix,
    ).reshape(2, n_bands, 3, n_nodes, 2 * shape[0], 2 * shape[1])
    n_pix = _lattice(
        jnp.ones((n_bands, 1, *shape)), response.kernels, response.radius_pix
    )[:, :, 0]

    def per_contrast(wavelength_nm):
        gain = instrument.primary.collecting_area_m2 * instrument.system_throughput(
            wavelength_nm
        )
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
        jnp.moveaxis(background, 2, 0),
        n_pix,
        local_zodi,
        sky_frame,
        reads=(
            "star",
            "dynamics.frame",
            *(() if local_zodi is None else ("local_zodi",)),
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
    sets the local zodi: the star's ecliptic latitude and longitude from the
    Sun at that time give the record's zodi model its sightline, and the
    radiance at each wavelength node scales the record's uniform-sky table.
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
    :func:`coronagraphoto.snr` or :func:`coronagraphoto.exposure_time`, or to
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

    here = lattice_index(record.roll_deg if roll_deg is None else roll_deg)
    star_leakage, local_zodi, exozodi = (
        _bilinear(record.background[i], *here) for i in range(3)
    )
    if roll_deg is not None:
        exozodi = _bilinear(record.background[2], *lattice_index(record.roll_deg))
    if record.local_zodi is not None:
        latitude_deg, longitude_deg = (None, None)
        frame = record.sky_frame
        if frame.ra_deg is not None:
            latitude_deg, longitude_deg = ecliptic_pointing(
                frame.ra_deg, frame.dec_deg, frame.epoch, time_d
            )
        local_zodi = local_zodi * record.local_zodi.photon_radiance(
            record.quadrature.nodes_nm,
            ecliptic_latitude_deg=latitude_deg,
            helioecliptic_longitude_deg=longitude_deg,
        )
    planet = contrast * _linear(
        record.planet, separation / record.separation_step_arcsec
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
