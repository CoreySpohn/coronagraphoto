"""Signed time-dependent speckle excess in the stellar leakage budget."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    centroid,
    exposure_for,
    make_path,
    make_system,
    with_instrument,
)
from hwoutils.constants import s2d
from hwoutils.epoch import Epoch
from hwoutils.numerics import roundoff
from hwoutils.precision import PrecisionError
from optixstuff import AbstractSpeckleField, FramePlan
from optixstuff.grids import (
    Frame,
    Image,
    Measure,
    PixelGrid,
    SkyOrientation,
    coverage,
)
from optixstuff.spectral import Band, Quadrature
from skyscapes.scene import Star

import coronagraphoto as cp
from coronagraphoto import photon_rates


class Halo(AbstractSpeckleField):
    """A finite halo evolving over days of its own clock, on its own grid."""

    grid: PixelGrid = eqx.field(
        static=True,
        default=PixelGrid.focal(8, 0.5, Frame(550.0, 6.0, "circumscribed")),
    )
    epoch: Epoch = eqx.field(static=True, default=Epoch(2451545.0))
    epoch_d: float = eqx.field(static=True, converter=float, default=3.0)
    level: float = 1.5

    def spectral_breaks_nm(self):
        """The halo is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        """A time-dependent flux fraction per native pixel."""
        data = (
            jnp.zeros((8, 8))
            .at[5, 6]
            .set(64e-10 * (self.level + jnp.cos(elapsed_s * s2d)))
        )
        return Image(
            data,
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )


@pytest.mark.float64_only(reason="speckle clock rebasing at nonzero epoch_d")
def test_speckle_leakage_day_clock():
    """Fails on epoch subtraction twice, wrong units or dead path wiring.

    Over exactly one reflected halo footprint the linear reconstruction
    conserves its native flux at 2:1 sampling, to 64 eps. A larger detector
    would include reflected replicas. Roll cannot rotate this instrumental halo.
    The halo grid is the detector footprint, so its outermost pixel ring is
    unsupported (the linear read needs half a source pixel of margin) and
    renders zero beside the base render; the excess is taken over the
    supported pixels of the halo run, where the halo has no flux (the lit
    pixel is interior), so the totals and the centroid are unchanged.
    """
    path = make_path(detector_n=16)
    wave = jnp.array([400.0, 500.0, 600.0, 700.0])
    times = jnp.array([-10.0, 20.0])
    flux = jnp.array([[1.0, 3.0]] * 4)
    system = make_system(
        star=Star(wavelengths_nm=wave, times_d=times, flux_density_jy=flux)
    )
    q = Quadrature.build(
        (Band(549.5, 550.5),), system, path.select("imager"), Halo(), order=1
    )
    scene = system

    def rates(scene, path, time_d, roll_deg=0, speckle=None):
        if speckle is not None:
            path = eqx.tree_at(
                lambda p: p.instruments[0].speckle,
                path,
                speckle,
                is_leaf=lambda v: v is None,
            )
        event, sampling = exposure_for(scene, q, time_d=time_d, roll_deg=roll_deg)
        return photon_rates(scene, path, event, sampling=sampling)

    def band_mean(star, time):
        """The star's mean over the one nm band: eight Gauss points of its own read."""
        x, w = np.polynomial.legendre.leggauss(8)
        values = star.photon_flux_density(550.0 + 0.5 * x, time, dist_pc=10.0)
        return float(jnp.sum(w / 2 * values))

    def excess(a, b):
        """The rate of ``a`` minus ``b`` on the pixels ``a`` supports."""
        return jnp.where(a.supported[:, None], a.star_leakage - b.star_leakage, 0.0)

    snapshots = []
    for time, factor in ((3.0, 2.5), (3.0 + np.pi, 0.5)):
        base = rates(scene, path, time)
        first = rates(scene, path, time, speckle=Halo())
        rolled = rates(scene, path, time, roll_deg=30, speckle=Halo())
        snapshots.append(first)
        if factor > 0:
            grid = PixelGrid.regular(
                path.instruments[0].detector.shape,
                float(path.instruments[0].detector.pixel_scale_arcsec),
                "arcsec",
            )
            np.testing.assert_allclose(
                centroid(excess(first, base)[0, 0], grid),
                [grid.center_x + 5.0, grid.center_y + 3.0],
                rtol=0,
                atol=1e-10,  # Exact 2:1 sampling, floating-point subtraction.
            )
        phi = band_mean(system.star, time)
        expected = 40 * 0.4 * phi * 64 * 1e-10 * factor
        np.testing.assert_allclose(
            excess(first, base).sum(),
            expected,
            rtol=64 * np.finfo(float).eps,
        )
        np.testing.assert_array_equal(first.star_leakage, rolled.star_leakage)

        # Same physical instant: the scene epoch and its samples move by the
        # shift, the event is stated on the shifted epoch, and the halo, which
        # is hardware, keeps its own clock.
        shift = 2.25
        shifted = make_system(
            star=Star(wavelengths_nm=wave, times_d=times - shift, flux_density_jy=flux),
            epoch=Epoch(2451547.0, 0.25),
            epoch_d=-shift,
        )
        same_date = rates(shifted, path, time - shift, speckle=Halo())
        for a, b in zip(
            jax.tree.leaves(first), jax.tree.leaves(same_date), strict=True
        ):
            # Rebased cubic interpolation/orbits incur roundoff; 1e-10 relative.
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-18)

    # A signed excess lowers the floor without clipping the total: -0.01 of
    # the halo spreads to at most ~2e-11 per detector pixel, under the 6.25e-11
    # fixture floor. Fails if the excess itself is clipped at zero. The sum
    # cancels against a floor ~250 times larger: 1e-10 relative.
    base = rates(scene, path, 3.0)
    dimmed = rates(scene, path, 3.0, speckle=Halo(level=-1.01))
    phi = band_mean(system.star, 3.0)
    np.testing.assert_allclose(
        excess(dimmed, base).sum(),
        40 * 0.4 * phi * 64e-10 * -0.01,
        rtol=1e-10,
    )

    vectorized = jax.vmap(lambda t: rates(scene, path, t, speckle=Halo()))(
        jnp.array([3.0, 3.0 + np.pi])
    )
    for index, expected in enumerate(snapshots):
        for a, b in zip(
            jax.tree.leaves(vectorized), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(a[index], b, rtol=1e-10, atol=1e-18)


def test_render_converts_scene_time_once_and_refuses_x32(precision):
    """Fails if the render ignores the field's epoch or subtracts it itself.

    Truth: the same halo anchored at zero and rendered at the elapsed time.
    Under x64 a halo at ``epoch_d = 2`` rendered at ``2 + t * s2d`` d gives the
    star leakage of the zero-anchored halo at ``t * s2d`` d to 4 float64 eps
    (the binary64 subtraction rounds by at most half an ulp of 2.7 d, which
    moves the cosine by under 1e-16 of the halo). Under x32 the anchored render
    raises ``PrecisionError`` and the zero-anchored one does not. Fails if the
    render passes ``time_d * d2s`` (the epoch is ignored) or forms
    ``(time_d - epoch_d) * d2s`` itself (the x32 render no longer raises).
    """
    t_s = 0.7 * 86400.0

    def render(time_d, speckle):
        path = make_path(detector_n=16)
        path = eqx.tree_at(
            lambda p: p.instruments[0].speckle,
            path,
            speckle,
            is_leaf=lambda v: v is None,
        )
        system = make_system()
        q = Quadrature.build(
            (Band(549.5, 550.5),), system, path.select("imager"), speckle, order=1
        )
        event, sampling = exposure_for(system, q, time_d=time_d)
        return photon_rates(system, path, event, sampling=sampling)

    zero = render(t_s * s2d, Halo(epoch_d=0.0))
    if precision == np.float64:
        anchored = render(2.0 + t_s * s2d, Halo(epoch_d=2.0))
        np.testing.assert_allclose(
            anchored.star_leakage, zero.star_leakage, rtol=4 * np.finfo(float).eps
        )
    else:
        with pytest.raises(PrecisionError, match="epoch_d"):
            render(2.0 + t_s * s2d, Halo(epoch_d=2.0))


def test_speckle_clock_is_the_hardware_clock_and_ignores_the_scene_epoch(precision):
    """Fails if the speckle field is fed the scene-epoch time.

    One non-constant halo (epoch ``E``) sits on one observatory. Two scenes
    differ only in their epoch (``E`` and ``E`` plus 10 d) and are rendered at
    one physical instant: one event stated on ``E``. Under x64 the stellar
    leakage, which carries the halo, is equal pixel for pixel, because the halo
    reads the event on its own epoch whatever the scene epoch is; and each
    planet sits where its own scene puts it at the scene's time (the shifted
    scene at the event equals the unshifted one 10 d earlier, and differs from
    the unshifted one at the event). A render that feeds the halo the scene's
    translation of the time changes the halo phase and fails the equality. Under
    x32 the unshifted scene renders and the shifted one raises
    ``PrecisionError``, because a binary32 time cannot be moved between epochs.
    """
    hardware = Epoch(2451545.0)
    halo = Halo(epoch_d=0.0)
    path = with_instrument(make_path(detector_n=16), speckle=halo)
    unshifted = make_system(separation_arcsec=0.02)
    shifted = make_system(separation_arcsec=0.02, epoch=Epoch(2451555.0))

    def render(scene, time_d):
        q = Quadrature.build(
            (Band(549.5, 550.5),), scene, path.select("imager"), halo, order=1
        )
        event, sampling = exposure_for(scene, q, time_d=time_d, epoch=hardware)
        return photon_rates(scene, path, event, sampling=sampling)

    here = render(unshifted, 3.5)
    if precision != np.float64:
        with pytest.raises(PrecisionError):
            render(shifted, 3.5)
        return
    there = render(shifted, 3.5)
    np.testing.assert_array_equal(there.star_leakage, here.star_leakage)
    earlier = render(unshifted, 3.5 - 10.0)
    np.testing.assert_allclose(there.planet, earlier.planet, rtol=1e-12, atol=0)
    assert not np.allclose(there.planet, here.planet, rtol=1e-3, atol=0)
    assert np.any(np.asarray(there.planet) > 0)


class Patch(AbstractSpeckleField):
    """A smooth excess on a square grid narrower than the detector.

    The excess density is ten times the static floor of the test coronagraph
    (1e-9 per square lambda/D), with a gentle modulation so that no two cells
    are equal.
    """

    grid: PixelGrid = eqx.field(static=True)
    reconstruction: str = eqx.field(static=True)
    data: jax.Array
    epoch: Epoch = eqx.field(static=True, default=Epoch(2451545.0))
    epoch_d: float = eqx.field(static=True, converter=float, default=0.0)

    def spectral_breaks_nm(self):
        """The excess is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        """The same cell integrals at every node and time."""
        return Image(
            self.data,
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
            self.reconstruction,
        )


_FRAME = Frame(550.0, 6.0, "circumscribed")


def _patch(n, cell_lod, reconstruction, crop=None):
    """A Patch of n cells; with ``crop``, the central ``crop`` cells of it."""
    grid = PixelGrid.focal(n, cell_lod, _FRAME)
    x, y = grid.coordinates()
    data = (
        1e-8 * cell_lod**2 * (1 + 0.5 * jnp.cos(0.8 * x + 0.2) * jnp.cos(0.6 * y - 0.1))
    )
    if crop is not None:
        start = (n - crop) // 2
        data = data[start : start + crop, start : start + crop]
        grid = PixelGrid.focal(crop, cell_lod, _FRAME)
    return Patch(grid, reconstruction, data)


# Detector pixel in field cells: (field cells, cell lod, detector pixels, pixel lod).
_GEOMETRY = {
    0.25: (8, 1.0, 64, 0.25),
    2.5: (16, 0.25, 13, 0.625),
    4.0: (32, 0.25, 16, 1.0),
}
_SAMPLING_MARGIN = {"linear": 0.5, "cubic": 2.0, "band_limited": 2.0}
_DEPOSIT_HALF_SUPPORT = {"linear": 1.0, "cubic": 2.0}


def _setup(pixel_cells, *, plan=None):
    """A static table that covers the detector, and a two-node band at 550 nm."""
    _, _, detector_n, pixel_lod = _GEOMETRY[pixel_cells]
    path = make_path(n=72, step=0.25, detector_n=detector_n, detector_step=pixel_lod)
    scene = make_system()
    exposure = cp.build_exposure(
        (Band(549.5, 550.5),),
        epoch=scene.epoch,
        instrument="imager",
        time_d=0.0,
        plan=plan,
    )
    sampling = cp.build_sampling(path, exposure, order=1, scene=scene)
    assert np.asarray(sampling.quadrature.nodes_nm).size == 2
    return scene, path, exposure, sampling


def _with(path, field):
    return eqx.tree_at(
        lambda p: p.instruments[0].speckle, path, field, is_leaf=lambda v: v is None
    )


def _field_coverage(field, path, nodes):
    """``coverage`` of the field's grid at every node, ANDed, from primitives."""
    detector = replace(
        path.instruments[0].detector.grid, orientation=SkyOrientation(0.0)
    )
    covered = np.ones(detector.shape, bool)
    for w in nodes:
        source = replace(field.grid.to_arcsec(w), orientation=detector.orientation)
        covered &= np.asarray(
            coverage(
                source,
                detector,
                reconstruction=field.reconstruction,
                measure=Measure.integral(),
            )
        )
    return covered


def _beyond_grown_footprint(field, path, nodes):
    """Pixels wholly outside the field footprint grown by its read margin.

    Geometry only: the footprint and the margin are taken at the node where
    each is largest, the margin as the read takes it (the sampling margin, or
    ``max(m, (k - 1/2) g - 1/2)`` cells when a pixel of g cells deposits).
    """
    detector = path.instruments[0].detector.grid
    pixel = float(path.instruments[0].detector.pixel_scale_arcsec)
    n, cell_lod = field.grid.shape[0], float(field.grid.step)
    reach = 0.0
    for w in nodes:
        cell = cell_lod * float(_FRAME.lod_to_arcsec(w))
        margin = _SAMPLING_MARGIN[field.reconstruction]
        g = pixel / cell
        if field.reconstruction in _DEPOSIT_HALF_SUPPORT and g >= 1:
            k = _DEPOSIT_HALF_SUPPORT[field.reconstruction]
            margin = max(margin, (k - 0.5) * g - 0.5)
        reach = max(reach, (n / 2 + margin) * cell)
    x, y = (np.asarray(c) for c in detector.coordinates())
    return (np.abs(x) - pixel / 2 >= reach) | (np.abs(y) - pixel / 2 >= reach)


@pytest.mark.parametrize(
    ("reconstruction", "pixel_cells"),
    [
        ("linear", 0.25),
        ("cubic", 0.25),
        ("band_limited", 0.25),
        ("linear", 2.5),
        ("cubic", 2.5),
        ("linear", 4.0),
        ("cubic", 4.0),
    ],
)
def test_static_support_and_the_static_light_beyond_the_field(
    reconstruction, pixel_cells
):
    """Fails if the static light beyond the field is kept by default or lost on request.

    A static table that covers the whole detector and a speckle field half the
    detector's width holding a smooth excess ten times the static floor, read
    linear, cubic or band-limited, on detector pixels of 0.25, 2.5 and 4 field
    cells, over a two-node band. Truth: the render with no field, and the
    field's ``coverage`` at each node from primitives; tolerance exact (the
    same arrays through the same program, or an added exact zero).

    Default render: ``static_supported`` equals the no-field ``supported``,
    ``supported`` equals ``static_supported`` AND the field coverage over the
    nodes, the product terms are exactly 0.0 where ``supported`` is False,
    and the record has the no-field record's tree structure. Keyword render:
    both flags equal the default's, the star leakage equals the default's
    where ``supported`` and the no-field render's on every pixel wholly
    outside the field footprint grown by the read margin at the node where
    each is largest, and it is positive on every ``static_supported`` pixel.
    Fails for a default that keeps static light, a keyword that changes a flag
    or a supported value, or a field whose excess reaches beyond its margin.
    """
    n, cell_lod, _, _ = _GEOMETRY[pixel_cells]
    field = _patch(n, cell_lod, reconstruction)
    scene, path, exposure, sampling = _setup(pixel_cells)
    nodes = np.asarray(sampling.quadrature.nodes_nm).ravel()
    alone = cp.expected(scene, path, exposure, sampling=sampling)
    default = cp.expected(scene, _with(path, field), exposure, sampling=sampling)
    kept = cp.expected(
        scene,
        _with(path, field),
        exposure,
        static_beyond_speckle=True,
        sampling=sampling,
    )

    static = np.asarray(alone.supported)[0]
    assert static.all(), "the static table must cover the detector"
    np.testing.assert_array_equal(alone.static_supported, alone.supported)
    np.testing.assert_array_equal(default.static_supported, alone.supported)
    strict = static & _field_coverage(field, path, nodes)
    assert strict.any() and not strict.all()
    np.testing.assert_array_equal(np.asarray(default.supported)[0], strict)
    for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
        np.testing.assert_array_equal(
            np.asarray(getattr(default, name))[0][~strict], 0.0
        )
    assert jax.tree.structure(default) == jax.tree.structure(alone)

    np.testing.assert_array_equal(kept.supported, default.supported)
    np.testing.assert_array_equal(kept.static_supported, default.static_supported)
    leak = np.asarray(kept.star_leakage)[0]
    np.testing.assert_array_equal(
        leak[strict], np.asarray(default.star_leakage)[0][strict]
    )
    beyond = _beyond_grown_footprint(field, path, nodes)
    assert beyond.any() and (~strict & ~beyond).any()
    np.testing.assert_array_equal(
        leak[beyond], np.asarray(alone.star_leakage)[0][beyond]
    )
    assert np.all(leak[static] > 0)


@pytest.mark.parametrize(
    ("reconstruction", "pixel_cells"), [("linear", 2.5), ("cubic", 4.0)]
)
def test_cropped_field_matches_the_wide_field_where_supported(
    reconstruction, pixel_cells
):
    """Guard: fails if the field's support margin is smaller than its read's reach.

    The same field sampled on a grid of twice its cells and on its pixel-aligned
    central half, on detector pixels of 2.5 and 4 cells, where the read deposits.
    Truth: the wide render, which reads no edge of the crop. Tolerance: 64 eps
    relative (the same samples through the same linear read; the sums differ in
    order only).
    """
    n, cell_lod, _, _ = _GEOMETRY[pixel_cells]
    scene, path, exposure, sampling = _setup(pixel_cells)
    wide = cp.expected(
        scene,
        _with(path, _patch(2 * n, cell_lod, reconstruction)),
        exposure,
        sampling=sampling,
    )
    crop = cp.expected(
        scene,
        _with(path, _patch(2 * n, cell_lod, reconstruction, crop=n)),
        exposure,
        sampling=sampling,
    )
    supported = np.asarray(crop.supported)[0]
    assert supported.any() and not supported.all()
    np.testing.assert_allclose(
        np.asarray(crop.star_leakage)[0][supported],
        np.asarray(wide.star_leakage)[0][supported],
        rtol=64 * np.finfo(float).eps,
        atol=0,
    )


@pytest.mark.slow(
    reason="beyond-field renders and detector sampler, about 21 s projected in CI"
)
def test_observe_draws_the_default_render_beyond_the_field():
    """Fails if a frame holds static light beyond the field, which it cannot flag.

    A field half the detector's width. With one key the frame from ``observe``
    equals ``detector.draw`` of the default ``expected``; where ``supported``
    is False that expectation holds no product term, so its total is exactly
    the thermal term; and the keyword render differs from it on at least one
    pixel beyond the field. Truth: the default ``expected``; exact (the same
    rates and key through the same draw). The option given as a NumPy or JAX
    boolean is the same request as a Python one. The keyword photon record carries
    the same flags and integrates to the keyword ``expected`` (1e-12 relative,
    summation order).
    """
    plan = FramePlan(frame_time_s=10.0, n_frames=3.0)
    scene, path, exposure, sampling = _setup(2.5, plan=plan)
    path = _with(path, _patch(16, 0.25, "linear"))
    key = jax.random.PRNGKey(0)
    frame = cp.observe(key, scene, path, exposure, sampling=sampling)
    default = cp.expected(scene, path, exposure, sampling=sampling)
    kept = cp.expected(
        scene, path, exposure, static_beyond_speckle=True, sampling=sampling
    )
    np.testing.assert_array_equal(
        frame.image.data,
        eqx.filter_jit(path.instruments[0].detector.draw)(key, default, plan),
    )
    off = ~np.asarray(default.supported)
    assert off.any()
    np.testing.assert_array_equal(
        np.asarray(default.total())[off], np.asarray(default.thermal)[off]
    )
    assert np.any(np.asarray(kept.total())[off] != np.asarray(default.total())[off])
    for flag in (np.True_, jnp.bool_(True)):
        again = cp.expected(
            scene, path, exposure, static_beyond_speckle=flag, sampling=sampling
        )
        np.testing.assert_array_equal(again.star_leakage, kept.star_leakage)
    # The spectral route keeps the same light under the same flags.
    photons = cp.photon_rates(
        scene, path, exposure, static_beyond_speckle=True, sampling=sampling
    )
    np.testing.assert_array_equal(photons.supported, kept.supported)
    np.testing.assert_array_equal(photons.static_supported, kept.static_supported)
    np.testing.assert_allclose(
        path.instruments[0].detector.electrons(photons).star_leakage,
        kept.star_leakage,
        rtol=1e-12,
        atol=0,
    )


class MaskedPatch(Patch):
    """A Patch whose array is wider than its data: zero-filled off ``mask``."""

    mask: jax.Array = None

    def data_mask(self):
        """Cells that hold data (a leaf, so traced inside the render)."""
        return self.mask

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        """The Patch values where the mask holds, exactly 0.0 elsewhere."""
        image = super().realize(wavelength_nm=wavelength_nm, elapsed_s=elapsed_s)
        return replace(image, data=jnp.where(self.mask, image.data, 0.0))


def test_masked_field_support_ends_at_its_data():
    """Fails if a consumer ignores a field's data mask.

    A 24 cell linear field on detector pixels of 2.5 cells that holds data
    only left of x = 0.1 and below y = 0.9 lambda/D (zero-filled elsewhere,
    mask False); the mask has no mirror symmetry, so a mask read flipped along
    either axis fails. Independent of ``coverage``, every supported pixel's
    center lies over a cell with data and every pixel that reads only cells
    without data lies over one without. Truth: ``coverage(valid=mask)``
    from primitives, at both nodes. ``supported`` is ``static_supported`` AND
    that coverage, so it is False and ``static_supported`` True on the pixels
    over the cells without data; the keyword render equals the no-field render
    exactly wherever the read reaches only cells without data (``coverage``
    with the complementary mask), where the zero fill adds an exact zero.
    """
    plain = _patch(24, 0.25, "linear")
    x, y = (np.asarray(c) for c in plain.grid.coordinates())
    mask = (x < 0.1) & (y < 0.9)
    field = MaskedPatch(plain.grid, "linear", plain.data, mask=jnp.asarray(mask))
    scene, path, exposure, sampling = _setup(2.5)
    nodes = np.asarray(sampling.quadrature.nodes_nm).ravel()
    alone = cp.expected(scene, path, exposure, sampling=sampling)
    default = cp.expected(scene, _with(path, field), exposure, sampling=sampling)
    kept = cp.expected(
        scene,
        _with(path, field),
        exposure,
        static_beyond_speckle=True,
        sampling=sampling,
    )

    detector = replace(
        path.instruments[0].detector.grid, orientation=SkyOrientation(0.0)
    )

    def covered(valid):
        out = np.ones(detector.shape, bool)
        for w in nodes:
            source = replace(field.grid.to_arcsec(w), orientation=detector.orientation)
            out &= np.asarray(
                coverage(
                    source,
                    detector,
                    reconstruction="linear",
                    measure=Measure.integral(),
                    valid=valid,
                )
            )
        return out

    static = np.asarray(default.static_supported)[0]
    np.testing.assert_array_equal(static, np.asarray(alone.supported)[0])
    strict = np.asarray(default.supported)[0]
    np.testing.assert_array_equal(strict, static & covered(mask))
    no_data = covered(~mask)
    assert no_data.any() and strict.any()
    assert not strict[no_data].any() and static[no_data].all()
    assert (strict != (static & covered(np.ones_like(mask)))).any()
    np.testing.assert_array_equal(
        np.asarray(kept.star_leakage)[0][no_data],
        np.asarray(alone.star_leakage)[0][no_data],
    )
    # The cell under each pixel center, from the grids alone, at each node.
    px, py = (np.asarray(c) for c in detector.coordinates())
    for w in nodes:
        column, row = field.grid.to_arcsec(w).unit_to_pixel(px, py)
        column, row = (
            np.clip(np.rint(np.asarray(c)), 0, 23).astype(int) for c in (column, row)
        )
        under = mask[row, column]
        assert under[strict].all() and not under[no_data].any()


class ChromaticRamp(Patch):
    """Affine speckle density whose amplitude grows linearly with wavelength."""

    def realize(self, *, wavelength_nm, elapsed_s=0.0):
        """Return the chromatic cell integrals in the field's own frame."""
        image = super().realize(wavelength_nm=wavelength_nm, elapsed_s=elapsed_s)
        return replace(image, data=image.data * wavelength_nm / 600.0)


def test_band_product_preserves_node_wavelength_for_speckles(gaussian_observation):
    """Compare the speckle excess to an affine field evaluated at each node.

    Linear reconstruction reproduces the ramp exactly. Its density times the
    pixel's node-dependent area gives its integral. Roundoff permits 128
    operations at the leakage scale before subtraction, covering the
    coordinate transforms, resampling and spectral products.
    """
    scene, path, exposure, sampling = gaussian_observation()
    frame = Frame(600, 6, "circumscribed")
    grid = PixelGrid.focal(24, 0.6, frame)
    x, _ = grid.coordinates()
    ramp = ChromaticRamp(grid, "linear", 1e-8 * (1 + 0.1 * x) * grid.pixel_area)
    base = cp.photon_rates(scene, path, exposure, sampling=sampling)
    result = cp.photon_rates(
        scene, with_instrument(path, speckle=ramp), exposure, sampling=sampling
    )
    assert np.all(result.supported)
    excess = np.asarray(result.star_leakage - base.star_leakage)[0]
    for got, wave in zip(excess, sampling.quadrature.nodes_nm[0], strict=True):
        pitch = float(path.instruments[0].detector.pixel_scale_arcsec) / float(
            frame.lod_to_arcsec(wave)
        )
        nx = path.instruments[0].detector.shape[1]
        centers = (np.arange(nx) - (nx - 1) / 2) * pitch
        gain = 20 * (0.4 + 0.004 * (float(wave) - 500))
        truth = gain * 1e6 * 1e-8 * (1 + 0.1 * centers) * pitch**2 * float(wave) / 600
        truth = np.broadcast_to(truth, got.shape)
        scale = float(result.star_leakage.max()) + float(base.star_leakage.max())
        np.testing.assert_allclose(
            got, truth, rtol=0, atol=roundoff(scale, n_ops=128, dtype=got.dtype)
        )
