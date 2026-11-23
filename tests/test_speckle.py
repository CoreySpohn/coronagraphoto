"""Signed time-dependent speckle excess in the stellar leakage budget."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import centroid, exposure_for, make_path, make_system
from hwoutils.epoch import Epoch
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
    """A finite halo evolving in days, on its own instrumental grid."""

    grid: PixelGrid = eqx.field(
        static=True,
        default=PixelGrid.focal(8, 0.5, Frame(550.0, 6.0, "circumscribed")),
    )
    epoch_d: float = 3.0
    level: float = 1.5

    def spectral_breaks_nm(self):
        """The halo is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, time_d=0.0):
        """A time-dependent flux fraction per native pixel."""
        data = (
            jnp.zeros((8, 8))
            .at[5, 6]
            .set(64e-10 * (self.level + jnp.cos(time_d - self.epoch_d)))
        )
        return Image(
            data,
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )


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
    q = Quadrature.build((Band(549.5, 550.5),), system, path, Halo(), order=1)
    scene = system

    def rates(scene, path, time_d, roll_deg=0, speckle=None):
        if speckle is not None:
            path = eqx.tree_at(
                lambda p: p.speckle, path, speckle, is_leaf=lambda v: v is None
            )
        return photon_rates(
            scene, path, exposure_for(scene, q, time_d=time_d, roll_deg=roll_deg)
        )

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
                path.detector.shape, float(path.detector.pixel_scale_arcsec), "arcsec"
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

        # Same physical epoch: shift orbit, stellar samples and speckle clock.
        shift = 2.25
        shifted = make_system(
            star=Star(wavelengths_nm=wave, times_d=times - shift, flux_density_jy=flux),
            epoch=Epoch(2451547.0, 0.25),
            epoch_d=-shift,
        )
        same_date = rates(
            shifted, path, time - shift, speckle=Halo(epoch_d=3.0 - shift)
        )
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


class Patch(AbstractSpeckleField):
    """A smooth excess on a square grid narrower than the detector.

    The excess density is ten times the static floor of the test coronagraph
    (1e-9 per square lambda/D), with a gentle modulation so that no two cells
    are equal.
    """

    grid: PixelGrid = eqx.field(static=True)
    reconstruction: str = eqx.field(static=True)
    data: jax.Array
    epoch_d: float = 0.0

    def spectral_breaks_nm(self):
        """The excess is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, time_d=0.0):
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
        scene, path, (Band(549.5, 550.5),), time_d=0.0, order=1, plan=plan
    )
    assert np.asarray(exposure.quadrature.nodes_nm).size == 2
    return scene, path, exposure


def _with(path, field):
    return eqx.tree_at(lambda p: p.speckle, path, field, is_leaf=lambda v: v is None)


def _field_coverage(field, path, nodes):
    """``coverage`` of the field's grid at every node, ANDed, from primitives."""
    detector = replace(path.detector.grid, orientation=SkyOrientation(0.0))
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
    detector = path.detector.grid
    pixel = float(path.detector.pixel_scale_arcsec)
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
    scene, path, exposure = _setup(pixel_cells)
    nodes = np.asarray(exposure.quadrature.nodes_nm).ravel()
    alone = cp.expected(scene, path, exposure)
    default = cp.expected(scene, _with(path, field), exposure)
    kept = cp.expected(scene, _with(path, field), exposure, static_beyond_speckle=True)

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
    scene, path, exposure = _setup(pixel_cells)
    wide = cp.expected(
        scene, _with(path, _patch(2 * n, cell_lod, reconstruction)), exposure
    )
    crop = cp.expected(
        scene, _with(path, _patch(2 * n, cell_lod, reconstruction, crop=n)), exposure
    )
    supported = np.asarray(crop.supported)[0]
    assert supported.any() and not supported.all()
    np.testing.assert_allclose(
        np.asarray(crop.star_leakage)[0][supported],
        np.asarray(wide.star_leakage)[0][supported],
        rtol=64 * np.finfo(float).eps,
        atol=0,
    )


def test_observe_draws_the_default_render_beyond_the_field():
    """Fails if a frame holds static light beyond the field, which it cannot flag.

    A field half the detector's width. With one key the frame from ``observe``
    equals ``detector.draw`` of the default ``expected``; where ``supported``
    is False that expectation holds no product term, so its total is exactly
    the thermal term; and the keyword render differs from it on at least one
    pixel beyond the field. Truth: the default ``expected``; exact (the same
    rates and key through the same draw). The keyword photon record carries
    the same flags and integrates to the keyword ``expected`` (1e-12 relative,
    summation order).
    """
    plan = FramePlan(frame_time_s=10.0, n_frames=3.0)
    scene, path, exposure = _setup(2.5, plan=plan)
    path = _with(path, _patch(16, 0.25, "linear"))
    key = jax.random.PRNGKey(0)
    frame = cp.observe(key, scene, path, exposure)
    default = cp.expected(scene, path, exposure)
    kept = cp.expected(scene, path, exposure, static_beyond_speckle=True)
    np.testing.assert_array_equal(
        frame.data, eqx.filter_jit(path.detector.draw)(key, default, plan)
    )
    off = ~np.asarray(default.supported)
    assert off.any()
    np.testing.assert_array_equal(
        np.asarray(default.total())[off], np.asarray(default.thermal)[off]
    )
    assert np.any(np.asarray(kept.total())[off] != np.asarray(default.total())[off])
    # The spectral route keeps the same light under the same flags.
    photons = cp.photon_rates(scene, path, exposure, static_beyond_speckle=True)
    np.testing.assert_array_equal(photons.supported, kept.supported)
    np.testing.assert_array_equal(photons.static_supported, kept.static_supported)
    np.testing.assert_allclose(
        path.detector.electrons(photons).star_leakage,
        kept.star_leakage,
        rtol=1e-12,
        atol=0,
    )
