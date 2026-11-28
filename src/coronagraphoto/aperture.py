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

from dataclasses import fields, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.epoch import Epoch
from hwoutils.leaves import is_traced
from hwoutils.numerics import COARSEST, eps
from hwoutils.precision import like, precision_of
from hwoutils.provenance import (
    Stamps,
    bind_reads,
    build_stamps,
    merge_stamps,
    prefix_stamps,
    qualify,
    require_stamps,
)
from jax.typing import ArrayLike
from jaxtyping import Array
from optixstuff import (
    Exposure,
    Observatory,
    Pointing,
    Rates,
    SelectedInstrument,
    SpectralSampling,
)
from optixstuff.apertures import DEFAULT_APERTURE_RULE, Circular
from optixstuff.coronagraph import AbstractImagingCoronagraph, MultiBandCoronagraph
from optixstuff.grids import Image, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Quadrature, Resolution
from orbix.system import SkyFrame
from skyscapes import AbstractLocalZodi, Scene

from coronagraphoto._version import __version__
from coronagraphoto.simulation import (
    _band_center_dtype,
    _host_child_index,
    _host_stage,
    _real_counts,
    _routed_children,
    _run,
)

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
        reads: Owner-qualified paths of the values the tables were measured
            from: the selected coronagraph (each child a band routes to, for a
            multiband product), the detector, the pupil and the quadrature
            nodes and weights. Changing one leaves the response stale; rebuild
            it. The throughput train is not read: the tables hold aperture
            fractions, and the collecting gain enters the record.
        built_at: Build facts of the hardware and layout the tables were
            measured for, as typed stamps. Admission compares them with the
            caller's view, also under ``jax.jit``; a hand-built response has
            none.
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
    _separation_limit_words: Array | None = None
    reads: tuple[str, ...] = eqx.field(static=True, default=(), kw_only=True)
    built_at: Stamps = eqx.field(static=True, default=(), kw_only=True)


class AperturePosition(eqx.Module):
    """Sky coordinates and ring interpolation prepared for one separation grid.

    Build with :meth:`ApertureRecord.prepare_position` before passing host
    coordinates through a JAX transformation. All leaves share the broadcast
    shape of the sky coordinates and can be mapped together with ``jax.vmap``.

    Attributes:
        east_arcsec: East offsets from the star [arcsec].
        north_arcsec: North offsets from the star [arcsec].
        lower_ring: Index of the lower contributing ring.
        near_upper: Whether the upper ring is the nearer knot.
        fraction: Distance from the nearer knot as a fraction of the ring step.
        between_rings: Whether the position lies between two tabulated rings.
        in_table: Whether the separation lies inside the tabulated range.
    """

    east_arcsec: Array
    north_arcsec: Array
    lower_ring: Array
    near_upper: Array
    fraction: Array
    between_rings: Array
    in_table: Array


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
        sampling: The wavelength rule and resolution the record was built on;
            ``quadrature`` and ``resolution`` read through it.
        pointing: Pointing the backgrounds were rendered at.
        build_time_d: Scene days (from ``epoch``) at which the record was built.
        pixel_scale_arcsec: Detector pixel pitch; the lattice pitch is half.
        separation_step_arcsec: Spacing of the planet table's separation axis.
        planet_response: Collecting gain times aperture throughput, multiplying
            a planet photon flux density, shape (2, W, Q, K).
        star: Node-effective stellar photon flux density at build, without
            collecting gain, shape (W, Q), in photons/s/m^2/nm.
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
        epoch: The scene epoch the record's clock counts from.
        reads: Owner-qualified paths of every value the record was built from
            (``scene.``, ``observatory.``, ``sampling.`` and ``exposure.``),
            each a whole leaf or a node covering the paths under it. Changing
            one of them leaves the record stale; rebuild it or hold the value.
            The planet's position and contrast, the read-time ``time_d`` and
            ``roll_deg`` and the local zodi's time are read live and are not
            listed.
        built_at: Build facts: the response's and the sampling's stamps under
            the prefixes ``response`` and ``sampling``, and the producer.
    """

    sampling: SpectralSampling
    pointing: Pointing
    build_time_d: Array
    pixel_scale_arcsec: Array
    separation_step_arcsec: Array
    planet_response: Array
    star: Array
    star_leakage: Array
    uniform_sky: Array
    exozodi: Array
    n_pix: Array
    unsupported: Array
    planet_unsupported: Array
    thermal: Array
    local_zodi: AbstractLocalZodi | None
    sky_frame: SkyFrame
    epoch: Epoch = eqx.field(static=True)
    reads: tuple[str, ...] = eqx.field(static=True)
    _separation_limit_words: Array | None = None
    built_at: Stamps = eqx.field(static=True, default=(), kw_only=True)

    @property
    def quadrature(self) -> Quadrature:
        """The wavelength nodes and weights the record was built on."""
        return self.sampling.quadrature

    @property
    def resolution(self) -> Resolution:
        """The resolution that reduces tabulated sources on ``quadrature``."""
        return self.sampling.resolution

    def prepare_position(
        self, east_arcsec: ArrayLike, north_arcsec: ArrayLike
    ) -> AperturePosition:
        """Condition a host position or batch for reads on this record's ring grid.

        Supply the original offsets in arcsec. The range and contributing rings
        are decided in binary64 and the distance from the nearer knot is formed
        before conversion to the record precision. Pass the result as the
        position argument of :func:`aperture_rates`, including under JIT.

        Args:
            east_arcsec: East offsets from the star [arcsec], host values.
            north_arcsec: North offsets from the star [arcsec], host values.

        Returns:
            The prepared position, with leaves of the broadcast input shape.
        """
        step = self.separation_step_arcsec
        last = self.planet_response.shape[-1] - 1
        limit = _separation_limit(step, self._separation_limit_words, last)
        east, north = np.broadcast_arrays(
            np.asarray(east_arcsec, dtype=np.float64),
            np.asarray(north_arcsec, dtype=np.float64),
        )
        separation = np.hypot(east, north)
        ring, upper, between, inside = _ring_segment(
            separation, limit / last, limit, last, np
        )
        fraction = _ring_fraction(
            separation, ring, upper, inside, limit / last, limit, last, np
        )
        return AperturePosition(
            jnp.asarray(east, dtype=step.dtype),
            jnp.asarray(north, dtype=step.dtype),
            jnp.asarray(ring),
            jnp.asarray(upper),
            jnp.asarray(fraction, dtype=step.dtype),
            jnp.asarray(between),
            jnp.asarray(inside),
        )


def _weights(rule, frame, shape, radius_pix, dx_pix, dy_pix, dtype=None):
    """Coverage of each pixel by the rule's circle, offset from the grid centre."""
    step = rule.radius_lod / radius_pix
    grid = PixelGrid.regular(shape, step, "lod", frame=frame)
    ones = Image(
        jnp.ones(shape, dtype=dtype),
        grid,
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )
    return np.asarray(rule.weights(ones, (dx_pix * step, dy_pix * step)))


def _ring_moments(
    model, detector, wavelengths_nm, band_center_nm, x_arcsec, y_arcsec, weights
):
    """Transfer host ring geometry at the wavelength rule's working precision."""
    center = jnp.asarray(band_center_nm)
    return _sample_ring(
        model,
        detector,
        wavelengths_nm,
        center,
        like(x_arcsec, center),
        like(y_arcsec, center),
        like(weights, center),
    )


@eqx.filter_jit
def _sample_ring(
    model, detector, wavelengths_nm, band_center_nm, x_arcsec, y_arcsec, weights
):
    """Aperture fractions (Q, 2, A) of A sources at each node of one band.

    Also returns the (Q, A) flags of sources inside the source domain. A source
    outside it has a zero image and so a zero fraction.
    """
    grid = detector.grid

    def node(wavelength_nm):
        native = model.at_wavelength(wavelength_nm).source_grid
        scale_nm = model.scale_wavelength_nm(wavelength_nm, band_center_nm)
        scale = native.frame.lod_to_arcsec(scale_nm)
        pixels = replace(
            grid.to_lod(scale_nm, native.frame), orientation=native.orientation
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
        return jnp.where(psfs.valid, moments, 0), psfs.valid

    return jax.lax.map(node, wavelengths_nm)


def _band_models(selected, quadrature, rule):
    """Per band: coronagraph model, lambda/D frame and aperture radius in pixels."""
    detector, coro = selected.detector, selected.coronagraph
    scale = float(detector.pixel_scale_arcsec)
    models, frames, radii = [], [], []
    for band in quadrature.bands:
        model = (
            coro.coronagraph_for_band(band)
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


def _node_model(coro, band_model, wavelength_nm):
    """Select a child per node only for dispatched single-wavelength products."""
    if isinstance(coro, MultiBandCoronagraph) and not coro.band_averaged:
        return coro.coronagraphs[_host_child_index(coro.coronagraphs, wavelength_nm)]
    return band_model


def _node_groups(coro, band_model, nodes_nm):
    """Group consecutive real nodes that read the same child."""
    groups = []
    for wave in np.asarray(nodes_nm):
        child = _node_model(coro, band_model, wave)
        if not groups or child is not groups[-1][0]:
            groups.append((child, []))
        groups[-1][1].append(wave)
    return [(child, np.asarray(nodes)) for child, nodes in groups]


def _probe_index(supported_row):
    """Index of the middle supported separation, or None."""
    inside = np.flatnonzero(np.asarray(supported_row))
    return None if inside.size == 0 else int(inside[inside.size // 2])


def _probe(
    model,
    selected,
    rule,
    frame,
    radius,
    nodes_nm,
    band_center_nm,
    supported,
    step,
    limit,
):
    """Aperture fraction of one source on the +x axis at the middle node of a band.

    ``nodes_nm`` are the band's real nodes, ``supported`` their ``(Q_real, K)``
    ring table. The host spacing and endpoint fix the source so that the
    build and check measure the same one.
    """
    mid = len(nodes_nm) // 2
    k = _probe_index(supported[mid])
    if k is None:
        return np.nan
    detector = selected.detector
    separation = limit if k == supported.shape[-1] - 1 else k * float(step)
    weights = _weights(
        rule,
        frame,
        detector.grid.shape,
        radius,
        separation / float(detector.pixel_scale_arcsec),
        0.0,
        band_center_nm.dtype,
    )
    moments, _ = _ring_moments(
        _node_model(selected.coronagraph, model, nodes_nm[mid]),
        detector,
        nodes_nm[mid : mid + 1],
        band_center_nm,
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


def _units(selected, quadrature):
    """Per coronagraph model a product reads: its owner path, label prefix and model.

    A single product has one unit. A multiband product has one per child the
    quadrature can reach (every child when the nodes are traced).
    """
    coro = selected.coronagraph
    path = selected.owner_path("coronagraph")
    if not isinstance(coro, MultiBandCoronagraph):
        return [(path, "coronagraph", coro)]
    return [
        (f"{path}.coronagraphs[{i}]", f"coronagraph.child{i}", coro.coronagraphs[i])
        for i in _routed_children(coro, quadrature)
    ]


def _model_stamps(model) -> Stamps:
    """A coronagraph's own build stamps: a delivery's, or a calibration's."""
    stamps = getattr(model, "built_at", None)
    if stamps is None and hasattr(model, "calibration_stamps"):
        stamps = model.calibration_stamps()
    return () if stamps is None else tuple(stamps)


def _hardware_stamps(selected, quadrature) -> Stamps:
    """Build facts of a selected view and a quadrature, from static data only."""
    facts = {
        "instrument": selected.name,
        "telescope_pupil_id": getattr(selected.pupil, "pupil_id", None),
        "layout": quadrature.layout_digest,
        "precision": str(precision_of(selected)),
    }
    units = _units(selected, quadrature)
    parts = []
    for _, prefix, model in units:
        if prefix == "coronagraph":
            facts["coronagraph_pupil_id"] = getattr(model, "pupil_id", None)
        else:
            facts[f"{prefix}.pupil_id"] = getattr(model, "pupil_id", None)
        parts.append(prefix_stamps(prefix, _model_stamps(model)))
    return merge_stamps(build_stamps(**facts), *parts)


def _frozen_reads(units, readers) -> tuple[str, ...]:
    """Observatory paths the cached readers of each coronagraph unit take."""
    out = []
    for path, _, model in units:
        frozen = getattr(model, "frozen_reads", None)
        for reader in readers if frozen is not None else ():
            out += bind_reads(frozen(reader), {"coronagraph": path})
    return tuple(out)


def _require_fresh(units, readers):
    """Refuse a cached read of a calibration that no longer matches its path.

    Host only: a unit whose leaves are traced is not checked.
    """
    for _, _, model in units:
        fresh = getattr(model, "require_fresh_calibration", None)
        if fresh is not None and _concrete(model):
            for reader in readers:
                fresh(reader=reader)


def _rule_stamp(rule) -> str:
    """The aperture rule as text: its type and static fields."""
    items = ", ".join(
        f"{f.name}={getattr(rule, f.name)!r}"
        for f in sorted(fields(rule), key=lambda f: f.name)
    )
    return f"{type(rule).__qualname__}({items})"


def _require_selected(selected):
    if not isinstance(selected, SelectedInstrument):
        raise TypeError(
            f"expected the view returned by Observatory.select(name), got "
            f"{type(selected).__name__}; pass observatory.select(name)"
        )


def check_response(
    response: "ApertureResponse", selected: SelectedInstrument, quadrature: Quadrature
) -> tuple[str, ...]:
    """Refuse a response built for something else, and report what was not checked.

    The static stage compares the response's build stamps with the facts of the
    caller's view and quadrature (configuration name, pupil identifiers, layout,
    precision and the coronagraph's own stamps). It runs under ``jax.jit``. A fact
    unknown on either side is left to the value stage. The value stage runs
    whenever the inputs are concrete, also when the stamps match: in order of
    cost, the wavelength nodes and bands, the aperture radius in pixels (which
    carries each band's lambda/D frame, the rule and the detector pixel scale),
    then one measured source per band, which carries the coronagraph's own
    tables. Under a trace it cannot run, so a node value or probe value that
    differs is not caught there: call this function on the host before a
    traced program and the pair is admitted once.

    Args:
        response: The response to check.
        selected: The view ``observatory.select(name)`` it will be read with.
        quadrature: The quadrature the record will be built on.

    Returns:
        The labels that could not be verified, sorted; ``"values"`` is among
        them when the value stage did not run.

    Raises:
        StampMismatchError: A recorded fact differs from the current one.
        ValueError: The values differ (wrong nodes, bands, radius or probe).
        TypeError: ``selected`` is not the view of ``Observatory.select``.
    """
    _require_selected(selected)
    current = _hardware_stamps(selected, quadrature)
    unverified = set(
        require_stamps(
            response.built_at,
            current,
            what="aperture response",
            labels=[label for label, _ in current],
        )
    )
    if not _concrete(selected, quadrature, response):
        return tuple(sorted(unverified | {"values"}))
    with jax.ensure_compile_time_eval():
        _compare_response(selected, quadrature, response)
    return tuple(sorted(unverified))


def _separation_limit(step, words, last):
    """Recover the unrounded host endpoint of a concrete separation grid."""
    pitch = float(step)
    if words is not None:
        limit = float(np.asarray(words).view(np.float64)[0])
        # An explicitly replaced pitch supersedes the original build geometry.
        if float(np.asarray(limit / last, dtype=step.dtype)) == pitch:
            return limit
    return last * pitch


def _compare_response(selected, quadrature, response):
    nodes, built = (
        np.asarray(quadrature.nodes_nm),
        np.asarray(response.quadrature.nodes_nm),
    )
    if nodes.shape != built.shape or not np.array_equal(nodes, built):
        raise ValueError("the response was built for other wavelength nodes")
    if quadrature.bands != response.quadrature.bands:
        raise ValueError("the response was built for other bands")
    models, frames, radii = _band_models(selected, quadrature, response.rule)
    counts = _real_counts(quadrature)
    # Allow eight float32 epsilons at each stored radius in pixels.
    radius_rtol = 8 * eps(COARSEST)
    if not np.allclose(
        radii, np.asarray(response.radius_pix), rtol=radius_rtol, atol=0
    ):
        raise ValueError(
            "the response was built for another detector pixel scale or "
            "lambda/D frame: its aperture radius is "
            f"{np.asarray(response.radius_pix).tolist()} pixels, this instrument's "
            f"is {radii}"
        )
    last = response.throughput.shape[-1] - 1
    limit = _separation_limit(
        response.separation_step_arcsec, response._separation_limit_words, last
    )
    dtype = _band_center_dtype(quadrature, selected)
    for w, (model, frame) in enumerate(zip(models, frames, strict=True)):
        expected = float(response.probe[w])
        measured = _probe(
            model,
            selected,
            response.rule,
            frame,
            float(response.radius_pix[w]),
            nodes[w, : counts[w]],
            np.asarray(quadrature.bands[w].center_nm, dtype=dtype),
            np.asarray(response.throughput_supported[w, : counts[w]]),
            limit / last,
            limit,
        )
        # Compare to 100 ppm of the probe fraction with roundoff at unit flux.
        if not np.isclose(
            measured, expected, rtol=1e-4, atol=eps(COARSEST), equal_nan=True
        ):
            raise ValueError(
                f"the response was built for another coronagraph or detector: "
                f"a source in band {w} puts {measured:.6g} of its light in the "
                f"aperture, the response recorded {expected:.6g}"
            )


def aperture_response(
    selected: SelectedInstrument,
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
    in an aperture centred on it. A band-averaged product uses the observing
    band's centre for every image. For single-wavelength children a node reads
    its covering child at the node scale. The aperture keeps the radius and
    frame of the band-centre child.

    ``quadrature`` must be the one the records will be built on, for example
    ``coronagraphoto.build_sampling(...).quadrature``. Only a
    ``Circular(radius_lod, centre="source")`` rule can be tabulated.

    The response remembers what it was built from: ``reads`` names the
    hardware values the tables were measured on and ``built_at`` stamps the
    configuration, pupil identifiers, quadrature layout, precision, rule and the
    coronagraph's own delivery. Editing that hardware afterwards leaves the
    response stale; rebuild it. The view is used for the call and not stored.

    Args:
        selected: The view ``observatory.select(name)`` whose detector and
            coronagraph the tables are built for.
        quadrature: The wavelength nodes and bands the tables are sampled at.
        rule: The aperture rule to sum with. Must be a ``Circular`` rule with
            ``centre="source"``.
        n_separations: Number of rings on the separation axis, at least two.
        n_angles: Number of sources on each ring, at least one.
        max_separation_arcsec: Outermost ring separation in arcsec. None uses
            the largest separation whose aperture stays on the detector.

    Returns:
        The tables of throughput, ring statistics and pixel weights for this
        instrument and band set, stored at the quadrature's working precision.

    Raises:
        TypeError: ``selected`` is not the view returned by ``select``.
    """
    _require_selected(selected)
    precision_of(selected)
    precision_of(quadrature)
    if not isinstance(rule, Circular) or rule.centre != "source":
        raise NotImplementedError(
            'aperture tables need a Circular rule with centre="source"'
        )
    if n_separations < 2 or n_angles < 1:
        raise ValueError("need at least two separations and one angle")
    detector = selected.detector
    grid = detector.grid
    scale = float(detector.pixel_scale_arcsec)
    models, frames, radii = _band_models(selected, quadrature, rule)
    if max_separation_arcsec is None:
        max_separation_arcsec = (min(grid.shape) / 2 - 1 - max(radii)) * scale
    if max_separation_arcsec <= 0:
        raise ValueError("the aperture does not fit on the detector")
    separations = np.linspace(
        0.0, max_separation_arcsec, n_separations, dtype=np.float64
    )

    nodes = quadrature.nodes_nm
    dtype = _band_center_dtype(quadrature, selected)
    slots = _padded_slots(quadrature)
    counts = _real_counts(quadrature)
    throughput = np.empty((2, *nodes.shape, n_separations))
    ring_std = np.empty((*nodes.shape, n_separations))
    ring_supported = np.empty((*nodes.shape, n_separations), bool)
    size = 2 * int(np.ceil(max(radii) + 1)) + 1
    kernels = np.empty((2, len(models), 2, 2, size, size))
    for w, (model, frame, radius) in enumerate(zip(models, frames, radii, strict=True)):
        for a, b in np.ndindex(2, 2):
            kernel = _weights(rule, frame, (size, size), radius, b / 2, a / 2, dtype)
            kernels[:, w, a, b] = kernel, kernel**2
        center_nm = np.asarray(quadrature.bands[w].center_nm, dtype=dtype)
        groups = _node_groups(selected.coronagraph, model, nodes[w, : counts[w]])
        for k, separation in enumerate(separations):
            angle = 2 * np.pi * (np.arange(n_angles) + (k * _GOLDEN) % 1) / n_angles
            x, y = separation * np.cos(angle), separation * np.sin(angle)
            weights = np.stack(
                [
                    _weights(rule, frame, grid.shape, radius, dx, dy, dtype)
                    for dx, dy in zip(x / scale, y / scale, strict=True)
                ]
            )
            # Only the real nodes are imaged; a padded slot repeats the first.
            reads = [
                _ring_moments(child, detector, waves, center_nm, x, y, weights)
                for child, waves in groups
            ]
            moments = np.concatenate([result[0] for result in reads])[slots[w]]
            inside = np.concatenate([result[1] for result in reads])[slots[w]]
            throughput[:, w, :, k] = moments.mean(axis=-1).T
            ring_std[w, :, k] = moments[:, 0].std(axis=-1)
            ring_supported[w, :, k] = inside.all(axis=-1)
    probe = [
        _probe(
            model,
            selected,
            rule,
            frame,
            radius,
            nodes[w, : counts[w]],
            np.asarray(quadrature.bands[w].center_nm, dtype=dtype),
            ring_supported[w, : counts[w]],
            separations[1],
            separations[-1],
        )
        for w, (model, frame, radius) in enumerate(
            zip(models, frames, radii, strict=True)
        )
    ]
    units = _units(selected, quadrature)
    # A live reader of the coronagraph (``source_psfs``) takes nothing frozen,
    # so a complete path adds no frozen read here.
    reads = qualify(
        "observatory",
        dict.fromkeys(
            (
                *(path for path, _, _ in units),
                selected.owner_path("detector"),
                selected.owner_path("pupil"),
                *_frozen_reads(units, ("source_psfs",)),
            )
        ),
    ) + qualify("sampling", ("quadrature.nodes_nm", "quadrature.weights_nm"))
    built_at = merge_stamps(
        _hardware_stamps(selected, quadrature),
        build_stamps(
            producer="coronagraphoto.aperture_response",
            producer_version=__version__,
            rule=_rule_stamp(rule),
            n_separations=n_separations,
            n_angles=n_angles,
        ),
    )
    return ApertureResponse(
        rule,
        quadrature,
        jnp.asarray(radii, dtype=dtype),
        jnp.asarray(separations[1], dtype=dtype),
        jnp.asarray(throughput, dtype=dtype),
        jnp.asarray(ring_std, dtype=dtype),
        jnp.asarray(ring_supported),
        jnp.asarray(kernels, dtype=dtype),
        jnp.asarray(probe, dtype=dtype),
        # Store the host endpoint in integer words so batching and canonical
        # precision casts retain it without creating a wide float leaf.
        jnp.asarray(separations[-1:].view(np.uint32)),
        reads=reads,
        built_at=built_at,
    )


def _on_detector(radius_pix, ny, nx):
    """Whether the aperture at each lattice position is whole on the detector.

    Shape (W, 2 ny, 2 nx): a band's aperture of ``radius_pix`` pixels at
    detector position (i / 2, j / 2) lies inside the pixel edges.
    """
    # Admission is structural. A host radius and its binary32 representation
    # must make the same decision at an exactly representable pixel edge.
    radius = radius_pix.astype(COARSEST)[:, None]
    inside = []
    for n in (ny, nx):
        positions = jnp.arange(2 * n, dtype=COARSEST) / 2
        inside.append((radius <= positions + 0.5) & (radius <= n - 0.5 - positions))
    return inside[0][:, :, None] & inside[1][:, None, :]


def _lattice(maps, kernels, radius_pix):
    """Sum per-pixel maps (W, B, ny, nx) in apertures, to (2, W, B, 2 ny, 2 nx).

    An aperture cut by the detector edge has no complete sum and reads zero.
    """
    n_bands, n_maps, ny, nx = maps.shape
    size = kernels.shape[-1]
    dtype = jnp.result_type(maps, kernels)
    maps, kernels = maps.astype(dtype), kernels.astype(dtype)
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
        bands.append(jnp.where(inside, sums, 0))
    return jnp.stack(bands, axis=1)


def aperture_record(
    scene: Scene,
    observatory: Observatory,
    exposure: Exposure,
    response: ApertureResponse,
    *,
    sampling: SpectralSampling,
) -> ApertureRecord:
    """Tabulate one target's aperture rates for one exposure.

    The star leakage (with the instrument's speckle field, if any), a uniform
    sky of unit radiance and the scene's disk are rendered once by the image
    path at the event's time (stated on the scene's epoch) and
    ``exposure.pointing``, without planets,
    then summed in the response's aperture at every half-pixel position. Only
    the real nodes of each band are rendered and summed; a padded slot repeats
    the band's first real node.
    The record keeps ``scene.local_zodi`` and the scene's sky frame (position
    and epoch), from which :func:`aperture_rates` computes the local-zodi
    radiance at the time of each read; nothing of the local zodi is fixed at
    the build time. A sightline-dependent local zodi on a scene with no sky
    position still builds a record; it raises a ``ValueError`` at the read, not
    here. The planet table is the response's throughput times the effective
    collecting area, independent of the star. A planet enters the read as an
    east offset, a north offset and a photon flux density, or a contrast
    hypothesis relative to the stellar photon flux density stored at build.

    ``response`` must have been built for this instrument and for
    ``sampling.quadrature``. A static stage first compares the response's build
    stamps with the hardware and sampling, also under ``jax.jit``, and raises
    ``StampMismatchError`` naming the label that differs. On concrete inputs the
    call then also refuses another one
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

    The record's ``reads`` lists what it was built from, owner-qualified. From
    the scene: the ``star`` (for leakage and ``record.star``) and the
    ``dynamics.frame`` (distance, position and epoch), plus the ``local_zodi``
    when there is one and the ``disk`` with its two midplane angles when there
    is one; the planets and their orbits are not read. From the observatory: the
    response's reads, the throughput trains that set the collecting gain, and
    the selected configuration's speckle field, warm emission and flux scale.
    From the event: ``exposure.time_d`` and ``exposure.pointing.roll_deg``. From
    the sampling: its own reads. ``built_at`` holds the response's and the
    sampling's stamps under those prefixes. A coronagraph that defines
    ``require_fresh_calibration`` is asked, once on the host, about the cached
    readers the record uses (``uniform_sky_map``).

    The scene's ``flux_scale`` leaves enter as follows. The star's and the
    disk's scales are part of the nodes the record reads, so they are fixed at
    the build: changing one leaves the record stale. The local zodi's scale is
    applied at each read, through the kept model. A planet's scale is not
    read here; it is carried by the planet photon flux density the caller
    passes, which :func:`scene_aperture_rates` takes from ``Sources.planets``.

    Args:
        scene: The target scene, with its star, disk and local zodi.
        observatory: The observatory the exposure is rendered with.
        exposure: The commanded event, whose time (on the scene's epoch) and
            ``pointing`` the star leakage, uniform sky and disk are rendered at.
        response: The aperture tables built for ``observatory.select`` of the
            event's instrument and for ``sampling.quadrature``.
        sampling: The wavelength rule built for this event.

    Returns:
        The tabulated aperture rates in photons/s/nm at the nodes of the
        exposure's quadrature, ready to read with :func:`aperture_rates`.

    Raises:
        NotImplementedError: The exposure has an explicit pointing offset.
        StampMismatchError: The response or the sampling was built for other
            hardware, or a cached read of a coronagraph is stale.
        ValueError: The response does not match the instrument or quadrature.
    """
    sel, coro, time_d, speckle_time_d = _host_stage(
        scene, observatory, exposure, sampling
    )
    quadrature = sampling.quadrature
    nodes = quadrature.nodes_nm
    check_response(response, sel, quadrature)
    units = _units(sel, quadrature)
    _require_fresh(units, ("uniform_sky_map",))
    images = _run(
        scene,
        sel,
        coro,
        exposure,
        (time_d, speckle_time_d),
        sampling,
        "photons",
        planets=False,
        uniform_sky=True,
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
    dtype = response.kernels.dtype
    n_pix = _lattice(
        jnp.ones((n_bands, 1, *shape), dtype=dtype),
        response.kernels,
        response.radius_pix,
    )[:, :, 0]
    # An aperture is unsupported when it leaves the detector or any pixel under
    # it is: the weighted count of unsupported pixels is positive.
    bad = _lattice(
        (~images.supported).astype(dtype)[:, None],
        response.kernels,
        response.radius_pix,
    )[0, :, 0]
    unsupported = ~_on_detector(response.radius_pix, *shape) | (bad > 0)

    gain = jax.vmap(sel.effective_area_m2)(nodes.ravel()).reshape(nodes.shape)
    build_time = like(time_d, response.kernels)
    star = scene.star.node_flux_density(
        quadrature,
        sampling.resolution,
        jnp.atleast_1d(build_time),
        dist_pc=scene.dist_pc,
    )[..., 0]
    reads = tuple(
        dict.fromkeys(
            (
                *qualify(
                    "scene",
                    (
                        "star",
                        "dynamics.frame",
                        *(() if scene.local_zodi is None else ("local_zodi",)),
                        *(
                            ()
                            if scene.disk is None
                            else (
                                "disk",
                                "midplane_inc_deg",
                                "midplane_position_angle_deg",
                            )
                        ),
                    ),
                ),
                *response.reads,
                *qualify(
                    "observatory",
                    (
                        *(path for path, _, _ in units),
                        *_frozen_reads(units, ("uniform_sky_map",)),
                        sel.owner_path("detector"),
                        sel.owner_path("pupil"),
                        "telescope.attenuating_elements",
                        f"{sel.owner_path()}.attenuating_elements",
                        sel.owner_path("speckle"),
                        sel.owner_path("warm_emission"),
                        sel.owner_path("flux_scale"),
                    ),
                ),
                *qualify("exposure", ("time_d", "pointing.roll_deg")),
                *sampling.reads,
            )
        )
    )
    built_at = merge_stamps(
        prefix_stamps("response", response.built_at),
        prefix_stamps("sampling", sampling.built_at),
        build_stamps(
            producer="coronagraphoto.aperture_record", producer_version=__version__
        ),
    )
    return ApertureRecord(
        sampling,
        exposure.pointing,
        build_time,
        like(sel.detector.pixel_scale_arcsec, response.separation_step_arcsec),
        response.separation_step_arcsec,
        gain[None, :, :, None] * response.throughput,
        like(star, response.kernels),
        star_leakage,
        uniform_sky,
        exozodi,
        n_pix,
        unsupported.astype(dtype),
        (~response.throughput_supported).astype(dtype),
        images.thermal[:, :, 0, 0],
        scene.local_zodi,
        scene.sky_frame,
        scene.epoch,
        reads,
        _separation_limit_words=response._separation_limit_words,
        built_at=built_at,
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
    fraction = position - k
    value = pair[..., 0] + fraction * (pair[..., 1] - pair[..., 0])
    return value, (position >= 0) & (position <= last)


def _ring_segment(separation, pitch, limit, last, xp):
    """Select the contributing rings using a single decision resolution."""
    inside = separation <= limit
    distance = xp.where(inside, separation, 0)
    lower = xp.clip(xp.floor(distance / pitch), 0, last - 1).astype(np.int32)
    # A quotient just below an integer can round onto the next knot.
    lower -= (lower.astype(distance.dtype) * pitch > distance).astype(np.int32)
    lo = lower.astype(distance.dtype) * pitch
    hi = xp.where(lower + 1 == last, limit, (lower + 1).astype(distance.dtype) * pitch)
    upper = hi - distance < distance - lo
    between = (lo < distance) & (distance < hi)
    return lower, upper, between, inside


def _ring_fraction(separation, lower, upper, inside, pitch, limit, last, xp):
    """Distance from the nearer knot in units of its interval, at working precision."""
    lo = lower.astype(separation.dtype) * pitch
    hi = xp.where(
        lower + 1 == last, limit, (lower + 1).astype(separation.dtype) * pitch
    )
    offset = xp.where(upper, hi - separation, separation - lo)
    # A point admitted at decision resolution can lie just beyond the knot at
    # working precision. Clamp there while retaining resolved interior offsets.
    return xp.where(inside & (offset >= 0), offset / (hi - lo), 0)


def _array_position(record, east, north):
    """Decide on float32 inputs and knots, then interpolate at working precision."""
    step = record.separation_step_arcsec
    east, north = like(east, step), like(north, step)
    last = record.planet_response.shape[-1] - 1
    pitch = step.astype(COARSEST)
    distance = jnp.hypot(east.astype(COARSEST), north.astype(COARSEST))
    ring, upper, between, inside = _ring_segment(
        distance, pitch, last * pitch, last, jnp
    )
    fraction = _ring_fraction(
        jnp.hypot(east, north), ring, upper, inside, step, last * step, last, jnp
    )
    return AperturePosition(east, north, ring, upper, fraction, between, inside)


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
    east_arcsec: ArrayLike | AperturePosition,
    north_arcsec: ArrayLike | None = None,
    *,
    contrast: ArrayLike | None = None,
    planet: ArrayLike | None = None,
    time_d: ArrayLike,
    roll_deg: ArrayLike | None = None,
) -> Rates:
    """Read one planet's aperture rates, in photons/s/nm, from a record.

    The planet sits ``east_arcsec`` east and ``north_arcsec`` north of the
    star, the order and convention of ``Scene.positions``. Pass exactly one
    of ``planet`` (photon flux density in photons/s/m^2/nm) and ``contrast``
    (a planet-to-star hypothesis relative to the star at the record's build).
    Each is scalar or broadcasts to (W, Q). A scene read should pass its
    node-effective planet flux density, ``Sources.planets``. The aperture is
    centered on the planet.

    Concrete host sky coordinates are differenced from the nearest ring
    knots before narrowing to the record precision. To retain this resolution
    for compiled or batched reads, pass ``record.prepare_position(east, north)``
    in place of the two coordinates. Prepare it on the host for the same ring
    grid. Raw JAX coordinates select rings and exact knots on a shared float32
    grid for admission. Between two supported rings, the interpolated rate
    retains the offset resolved at record precision.

    ``time_d`` is the observation time in days from the record's epoch. It
    sets the local zodi: the record's zodi model is evaluated for the record's
    sky frame at that time, and its radiance at each wavelength node scales
    the record's uniform-sky table.
    It has no default, because the time a record was built at is not the time
    of a later read and the zodi would silently be that of the wrong season.
    The star leakage, ``record.star``, the disk and any speckle realization
    stay those of the exposure the record was built from.

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

    All arguments are scalars apart from ``contrast`` and ``planet``. Batch with
    ``jax.vmap``; a ``time_d`` shared by a batch of planets (``in_axes`` None,
    or closed over) costs one radiance evaluation for the whole batch.

    Args:
        record: The aperture record built for the target and exposure.
        east_arcsec: Planet offset east of the star in arcsec, or a prepared position.
        north_arcsec: Planet offset north of the star, in arcsec.
        contrast: Planet-to-star hypothesis relative to ``record.star``, scalar
            or broadcastable to (W, Q). Mutually exclusive with ``planet``.
        planet: Planet photon flux density in photons/s/m^2/nm, scalar or
            broadcastable to (W, Q). Mutually exclusive with ``contrast``.
        time_d: Observation time in days from the record's epoch.
        roll_deg: Position angle of detector +y for this read, in degrees.
            None uses the record's roll.

    Returns:
        The planet, star leakage, local zodi, exozodi and thermal rates in
        photons/s/nm, with ``n_pix``, ``variance`` and the ``supported`` flag.
    """
    if (contrast is None) == (planet is None):
        raise ValueError("give exactly one of contrast and planet")
    if isinstance(east_arcsec, AperturePosition):
        if north_arcsec is not None:
            raise ValueError("omit north_arcsec with a prepared position")
        step = record.separation_step_arcsec
        position = replace(
            east_arcsec,
            east_arcsec=like(east_arcsec.east_arcsec, step),
            north_arcsec=like(east_arcsec.north_arcsec, step),
            fraction=like(east_arcsec.fraction, step),
        )
    elif north_arcsec is None:
        raise ValueError("give both east_arcsec and north_arcsec")
    elif (
        not isinstance(east_arcsec, jax.Array)
        and not isinstance(north_arcsec, jax.Array)
        and _concrete(record.separation_step_arcsec, record._separation_limit_words)
    ):
        position = record.prepare_position(east_arcsec, north_arcsec)
    else:
        position = _array_position(record, east_arcsec, north_arcsec)
    east, north = position.east_arcsec, position.north_arcsec
    ny, nx = record.n_pix.shape[-2] // 2, record.n_pix.shape[-1] // 2

    def lattice_index(roll):
        x, y = SkyOrientation(roll).sky_to_detector(east, north)
        return (
            (ny - 1) + 2 * y / record.pixel_scale_arcsec,
            (nx - 1) + 2 * x / record.pixel_scale_arcsec,
        )

    built_roll = record.pointing.roll_deg
    fixed = lattice_index(built_roll)
    here = fixed if roll_deg is None else lattice_index(like(roll_deg, built_roll))
    star_leakage, inside_here = _bilinear(record.star_leakage, *here)
    local_zodi, _ = _bilinear(record.uniform_sky, *here)
    exozodi, inside_fixed = _bilinear(record.exozodi, *fixed)
    n_pix, _ = _bilinear(record.n_pix, *here)
    bad_here, _ = _bilinear(record.unsupported, *here)
    bad_fixed, _ = _bilinear(record.unsupported, *fixed)
    if record.local_zodi is not None:
        radiance = record.local_zodi.node_radiance(
            record.quadrature,
            record.resolution,
            record.sky_frame,
            like(time_d, record.planet_response),
        )
        local_zodi = local_zodi * like(radiance, record.planet_response)
    ring = position.lower_ring
    pair = jax.lax.dynamic_slice_in_dim(record.planet_response, ring, 2, axis=-1)
    near = jnp.where(position.near_upper, pair[..., 1], pair[..., 0])
    far = jnp.where(position.near_upper, pair[..., 0], pair[..., 1])
    ring_mask = jax.lax.dynamic_slice_in_dim(
        record.planet_unsupported, ring, 2, axis=-1
    )
    # A supported knot beside a missing ring reads only its own value.
    fraction = jnp.where(jnp.any(ring_mask != 0, axis=-1), 0, position.fraction)
    planet_response = near + fraction * (far - near)
    selected = planet if contrast is None else contrast
    selected = like(selected, record.planet_response)
    name = "planet" if contrast is None else "contrast"
    try:
        selected = jnp.broadcast_to(selected, record.star.shape)
    except ValueError:
        raise ValueError(
            f"{name} has shape {selected.shape}, which does not broadcast to "
            f"the record's (W, Q) = {record.star.shape}"
        ) from None
    if contrast is not None:
        selected = like(record.star, record.planet_response) * selected
    planet = selected * planet_response
    bad_ring = (
        (~position.near_upper | position.between_rings) & (ring_mask[..., 0] != 0)
    ) | ((position.near_upper | position.between_rings) & (ring_mask[..., 1] != 0))
    supported = (
        inside_here
        & inside_fixed
        & position.in_table
        & (bad_here == 0)
        & (bad_fixed == 0)
        & ~jnp.any(bad_ring, axis=-1)
    )

    def zero(x):
        return jnp.where(supported.reshape(supported.shape + (1,) * (x.ndim - 1)), x, 0)

    def rates(moment, variance=None):
        return Rates(
            record.quadrature,
            "photon",
            planet=zero(planet[moment]),
            star_leakage=zero(star_leakage[moment]),
            local_zodi=zero(local_zodi[moment]),
            exozodi=zero(exozodi[moment]),
            thermal=zero(record.thermal * n_pix[moment][:, None]),
            stellar_companions=jnp.zeros((), dtype=planet.dtype),
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
    photon flux density from ``Sources.planets`` on the record's quadrature
    and resolution. No division by the stellar flux enters the planet read.
    Nothing is rebuilt. The scene should be the one the record was built from
    (``record.reads`` lists what the record holds fixed).

    ``time_d`` has shape ``(T,)``, days from the scene's epoch, which must equal
    the record's epoch (a different epoch is refused, not reinterpreted);
    ``roll_deg`` is as in :func:`aperture_rates`. The returned ``Rates`` follows
    the layout of every ``Rates`` with a quadrature: the band and node axes lead and the
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
            Concrete dates keep their host precision for the scene clock.
        roll_deg: Position angle of detector +y, in degrees, as in
            :func:`aperture_rates`. None uses the record's roll.

    Returns:
        The rates of every planet at every time in photons/s/nm, with the
        shapes given above.
    """
    # The scene owns its sample origin and must subtract it from host dates
    # before they reach JAX. Lists of traced dates instead follow JAX promotion.
    if not isinstance(time_d, jax.Array) and not is_traced(time_d):
        device_dates = any(
            isinstance(t, jax.Array) or is_traced(t) for t in jax.tree.leaves(time_d)
        )
        time_d = (
            jnp.asarray(time_d)
            if device_dates or hasattr(time_d, "__jax_array__")
            else np.asarray(time_d)
        )
    if time_d.ndim != 1:
        raise ValueError("time_d must have shape (T,)")
    if scene.batch_shape:
        raise ValueError("scene_aperture_rates reads one system; vmap it over a batch")
    if scene.epoch != record.epoch:
        raise ValueError(
            "the scene's epoch differs from the epoch the record's clock counts "
            f"from ({scene.epoch!r} vs {record.epoch!r}); rebuild the record"
        )
    east, north = scene.positions(time_d)
    planet = scene.sources(record.quadrature, record.resolution, time_d).planets

    def read(east_arcsec, north_arcsec, planet, time):
        rates = aperture_rates(
            record,
            east_arcsec,
            north_arcsec,
            planet=planet,
            time_d=time,
            roll_deg=roll_deg,
        )
        return tuple(
            (
                *(getattr(r, name) for name in _READ_FIELDS),
                r.n_pix,
                r.supported,
            )
            for r in (rates, rates.variance)
        )

    times = None if record.local_zodi is None else jnp.asarray(time_d)
    over_times = jax.vmap(
        read, in_axes=(0, 0, -1, None if times is None else 0), out_axes=-1
    )
    mean, variance = jax.vmap(over_times, in_axes=(0, 0, -2, None), out_axes=-2)(
        east, north, planet, times
    )

    def rates(values, variance=None):
        *fields, n_pix, supported = values
        fields = [jnp.where(scene.valid[:, None], v, 0) for v in fields]
        return Rates(
            record.quadrature,
            "photon",
            **dict(zip(_READ_FIELDS, fields, strict=True)),
            stellar_companions=jnp.zeros((), dtype=fields[0].dtype),
            n_pix=jnp.where(scene.valid[:, None], n_pix, 0),
            variance=variance,
            supported=supported,
        )

    return rates(mean, rates(variance))
