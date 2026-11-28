"""Closure of the aperture record, S/N and exposure time against image sums."""

from functools import cache

import equinox as eqx
import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from conftest import closure_response, closure_system, make_path, write_yip
from optixstuff import Detector, FramePlan
from optixstuff.apertures import Circular
from optixstuff.grids import Image, Measure, SkyOrientation
from optixstuff.spectral import Band
from yippy import open_yip

import coronagraphoto as cp

TERMS = ("planet", "star_leakage", "local_zodi", "exozodi")
pytestmark = [
    pytest.mark.medium(
        reason="YIP load, scene operator and four renders at two detector pitches"
    ),
    pytest.mark.float64_only(reason="V&V reference values for calculator closure"),
]


def _bilinear_bound(table, row, column):
    """Bound the bilinear read of the last two axes at a fractional index.

    For a smooth function on a unit lattice the bilinear error is at most one
    eighth of the sum of the second derivatives along the two axes. Both are
    measured as the largest second differences over the four-by-four
    neighbourhood of the cell.
    """
    i, j = int(np.floor(row)), int(np.floor(column))
    patch = np.asarray(table[..., i - 1 : i + 3, j - 1 : j + 3])
    along_rows = abs(np.diff(patch, n=2, axis=-2)).max(axis=(-2, -1))
    along_columns = abs(np.diff(patch, n=2, axis=-1)).max(axis=(-2, -1))
    return (along_rows + along_columns) / 8


@pytest.fixture(scope="module")
def build_closure(tmp_path_factory):
    """Read one planet from the record and sum the same aperture on images.

    The scene is a Gaussian YIP (sigma 0.7 lambda/D, integral 0.3, pitch 0.5
    lambda/D, tables 8 lambda/D from the centre so that every pixel of the
    9.6 lambda/D fields is inside them at both band edges) with a parametric
    disk, a local zodi and one planet, observed in two bands at a 15 degree
    roll on 0.2 and 0.1 lambda/D pixels. No response is mocked or renormalized.
    """
    rule = Circular(0.7)
    coro = (
        open_yip(
            write_yip(
                tmp_path_factory.mktemp("calculator-closure") / "yip",
                reach=8,
                half_width=8,
            )
        )
        # The calculator consumes the image response only.
        .prepare(rules=(rule,), leakage_rule="aperture_mean")
        .with_scene_operator()
    )
    base = eqx.tree_at(lambda p: p.instruments[0].coronagraph, make_path(), coro)
    scale = float(coro.frame.lod_to_arcsec(550.0))
    system = closure_system(scale)
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    roll, contrast = 15.0, 1e-9
    # The circular face-on orbit starts at its 250 degree node. Use the
    # prescribed sky position for the aperture, independently of the renderer.
    position_angle = np.deg2rad(250.0)
    separation = 2.3 * scale
    east = separation * np.sin(position_angle)
    north = separation * np.cos(position_angle)
    x, y = SkyOrientation(roll).sky_to_detector(east, north)

    @cache
    def build(n, pitch, compact):
        """Build the nominal image and record for one detector sampling."""
        detector = Detector(
            shape=(n, n),
            pixel_scale_arcsec=pitch * scale,
            qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
            survival_fraction=0.9,
            dark_current_e_per_pixel_s=3e-4,
            clock_induced_charge_e_per_pixel_frame=0.005,
            read_noise_e_rms=0.2,
        )
        path = eqx.tree_at(lambda p: p.instruments[0].detector, base, detector)
        # One band with two nodes still exercises chromatic detector yield.
        # Nightly keeps both bands, the field-based rule and the ring spacing.
        observing_bands = bands[:1] if compact else bands
        options = dict(field_lod=0.0, order=2) if compact else {}
        exposure = cp.build_exposure(
            observing_bands,
            epoch=system.epoch,
            instrument="imager",
            time_d=0.0,
            roll_deg=roll,
        )
        sampling = cp.build_sampling(path, exposure, scene=system, **options)
        if compact:
            response = cp.aperture_response(
                path.select("imager"),
                sampling.quadrature,
                rule=rule,
                n_separations=7,
                n_angles=8,
                max_separation_arcsec=3.25 * scale,
            )
        else:
            response = closure_response(
                path, sampling.quadrature, rule, separation, outer_neighbours=1
            )
        record = cp.aperture_record(system, path, exposure, response, sampling=sampling)
        rates = cp.aperture_rates(record, east, north, contrast=contrast, time_d=0.0)
        electrons = detector.electrons(rates)

        # The independent reader: the rule's own weights on the expected images.
        images = cp.expected(system, path, exposure, sampling=sampling)
        weights = []
        for band in observing_bands:
            frame = coro.at_wavelength(band.center_nm).source_grid.frame
            lod = float(frame.lod_to_arcsec(band.center_nm))
            ones = Image(
                jnp.ones((n, n)),
                detector.grid.to_lod(band.center_nm, frame),
                (),
                Measure.integral(),
                "stellar_flux_fraction",
            )
            weights.append(rule.weights(ones, (float(x) / lod, float(y) / lod)))
        weights = jnp.stack(weights)

        def moments(source, weights=weights):
            return np.stack(
                [
                    [jnp.sum(getattr(source, t) * weights**m, (-2, -1)) for t in TERMS]
                    for m in (1, 2)
                ]
            )

        image = moments(images)
        calculated = np.stack(
            [[getattr(r, t) for t in TERMS] for r in (electrons, electrons.variance)]
        )
        pixels = np.stack([(weights**m).sum((-2, -1)) for m in (1, 2)])

        # Interpolation bands of the read, from the tables themselves.
        position = float(separation / response.separation_step_arcsec)
        knot = int(position)
        table = np.asarray(response.throughput[0])
        scatter = np.asarray(response.throughput_ring_std)[..., knot : knot + 2]
        curvature = abs(np.diff(table[..., knot - 1 : knot + 3], n=2, axis=-1)) / 8
        planet_band = (5 * scatter.max(-1) + curvature.max(-1)) / table[
            ..., knot : knot + 2
        ].min(-1)
        row = (n - 1) + 2 * float(y) / (pitch * scale)
        column = (n - 1) + 2 * float(x) / (pitch * scale)
        lattice_band = (
            _bilinear_bound(record.exozodi[0], row, column) / rates.exozodi
        ).max(-1)
        return dict(
            calculated=calculated,
            image=image,
            error=calculated / image - 1,
            pixels=pixels,
            pixel_ratio=np.stack([electrons.n_pix, electrons.variance.n_pix]) / pixels,
            planet_band=planet_band.max(-1),
            lattice_band=lattice_band,
            system=system,
            path=path,
            exposure=exposure,
            sampling=sampling,
            response=response,
            rates=rates,
            electrons=electrons,
            weights=weights,
            offsets=(east, north),
            contrast=contrast,
        )

    yield build
    build.cache_clear()


_COARSE = (48, 0.2)
_FINE = (96, 0.1)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((*_COARSE, True), id="0.2-lod-compact"),
        pytest.param(
            (*_COARSE, False),
            id="0.2-lod",
            marks=pytest.mark.slow(
                reason="48x48 aperture record with the full original nodes, ~35 s in CI"
            ),
        ),
        pytest.param(
            (*_FINE, False),
            id="0.1-lod",
            marks=pytest.mark.slow(
                reason="96x96 aperture record and image renders, ~35 s projected in CI"
            ),
        ),
    ],
)
def closure(request, build_closure):
    """Share one set of nominal reads across every check of this pitch."""
    return build_closure(*request.param)


@vv.case(
    "etc-count-rates",
    "code-verification",
    srq="planet count rate Cp; stellar leakage CRbs",
)
def test_planet_and_leakage_closure(closure):
    """Fails if the planet table or the leakage lattice disagrees with the images.

    Both readers use the same coronagraph products and the same exact pixel
    weights, so the planet differs only by pixel phase, which the ring mean
    averages, and by linear interpolation between rings. Basis: five ring
    standard deviations plus one eighth of the largest neighbouring second
    difference of the table; measured relative errors are 1.9e-3 on 0.2
    lambda/D pixels and 2.3e-4 on 0.1, and the band must stay below 5e-3.
    The leakage of this YIP is uniform, so every aperture sum is the same
    number and only float64 rounding of the convolution remains: 1e-9.
    """
    c = closure
    planet = abs(c["error"][0, 0])
    assert np.all(c["planet_band"] < 5e-3)
    assert np.all(planet <= c["planet_band"])
    assert np.all(abs(c["error"][0, 1]) <= 1e-9)
    assert np.all(c["calculated"][0, :2] > 0)


@vv.case("etc-count-rates", "code-verification", srq="planet count rate Cp")
def test_survival_applied_once(closure):
    """Fails on duplicate survival: the mutant is 0.9*C, 10% low.

    Chromatic QE and survival=0.9 stay on the real detector path of both
    readers. Basis: the planet band of the closure test; the mutant must lie
    outside it.
    """
    c = closure
    image, calculated = c["image"][0, 0], c["calculated"][0, 0]
    assert np.all(abs(calculated / image - 1) <= c["planet_band"])
    assert np.all(abs(0.9 * calculated / image - 1) > c["planet_band"])


@vv.case("etc-count-rates", "code-verification", srq="zodi and exozodi rates")
def test_background_closure(closure):
    """Fails on an omitted disk, wrong solid angle or a misplaced lattice read.

    The local zodi is uniform, so its aperture sum closes to float64 rounding
    of the convolution, 1e-9. The exozodi varies across the aperture
    positions: the bilinear read between half-pixel positions is bounded by
    one eighth of the summed second differences of the lattice around the
    planet. The bound is doubled because it is estimated from differences,
    not derived: 4.2e-4 on 0.2 lambda/D pixels and 1.1e-4 on 0.1, against
    measured errors of 1.3e-4 and 4.9e-5.
    Doubling the dust doubles both readers to a 1e-12 arithmetic allowance,
    and both must give a nonzero exozodi.
    """
    c = closure
    system, path, exposure = c["system"], c["path"], c["exposure"]
    sampling = c["sampling"]
    doubled = eqx.tree_at(
        lambda s: s.disk.amplitude.level, system, 2 * system.disk.amplitude.level
    )
    record = cp.aperture_record(
        doubled, path, exposure, c["response"], sampling=sampling
    )
    rates = cp.aperture_rates(record, *c["offsets"], contrast=c["contrast"], time_d=0.0)
    images = cp.expected(doubled, path, exposure, sampling=sampling)
    np.testing.assert_allclose(rates.exozodi / c["rates"].exozodi, 2.0, rtol=1e-12)
    dust_images = jnp.sum(images.exozodi * c["weights"]) / c["image"][0, 3].sum()
    np.testing.assert_allclose(dust_images, 2.0, rtol=1e-12)
    assert np.all(c["calculated"][0, 2:] > 0) and np.all(c["image"][0, 2:] > 0)
    assert np.all(abs(c["error"][0, 2]) <= 1e-9)
    exozodi_band = 2 * c["lattice_band"] + 1e-9
    assert np.all(exozodi_band < 1e-3)
    assert np.all(abs(c["error"][0, 3]) <= exozodi_band)


@vv.case("etc-count-rates", "code-verification", srq="aperture variance rates")
def test_squared_weight_closure(closure):
    """Fails if the w**2 tables or pixel counts misstate the variance rates.

    The summed weights are exact at every position (the circle's area), so
    the mean pixel count closes to 1e-9. The squared weights depend on pixel
    phase, and their half-pixel lattice is read bilinearly. Basis: measured;
    the errors are 3.8e-3 on 0.2 lambda/D pixels and 1.9e-3 on 0.1 for the
    backgrounds and pixel count, and 1.6e-2 and 3.4e-3 for the planet, whose
    ring mean also averages pixel phase. Held to 1e-2 and 3e-2, and each must
    shrink with the pixels.
    """
    c = closure
    np.testing.assert_allclose(c["pixel_ratio"][0], 1.0, rtol=1e-9)
    backgrounds = abs(c["error"][1, 1:]).max()
    pixels = abs(c["pixel_ratio"][1] - 1).max()
    planet = abs(c["error"][1, 0]).max()
    assert backgrounds < 1e-2 and pixels < 1e-2
    assert planet < 3e-2


@pytest.mark.slow(
    reason="both aperture records and matched image sums, ~72 s projected in CI"
)
@vv.case(
    "etc-count-rates",
    "code-verification",
    srq="planet count rate Cp and aperture variance rates",
)
def test_aperture_errors_decrease_with_finer_pixels(build_closure):
    """Fails if finer pixels leave more mean planet or squared-weight error.

    Truth and tolerances are the two image closures above: finer pixels leave
    less pixel phase for the ring mean to hide. Compare the planet mean in
    each band and the largest squared-weight error of each source group.
    """
    coarse, fine = (abs(build_closure(*p, False)["error"]) for p in (_COARSE, _FINE))
    assert np.all(fine[0, 0] < coarse[0, 0])
    assert fine[1, 1:].max() < coarse[1, 1:].max()
    assert fine[1, 0].max() < coarse[1, 0].max()


@vv.case("etc-inversion", "code-verification", srq="S/N at fixed time; time to S/N")
def test_snr_and_exposure_time_closure(closure):
    """Fails if the analytic S/N or time disagrees with the image budget.

    The image S/N sums planet, source, dark, clock-induced and read-noise
    variance over the same aperture through ``detector.mean_and_variance``
    and adds the residual squares, for 360 frames of 100 s. The analytic
    time is solved for that image S/N and must return the simulated live
    time. Basis: first-order propagation of the measured rate differences,
    S/N as p / sqrt(b) and time as b / p**2, with the planet error dp and
    the largest squared-weight error db: dp + db / 2 for S/N and 2 dp + db
    for the time, doubled for the terms the first order drops.
    """
    c = closure
    detector = c["path"].instruments[0].detector
    image, pixels, electrons = c["image"], c["pixels"], c["electrons"]
    pp = ox.PostProcessing(1e-4, 0.01, background_variance_factor=1.0)
    frame_time = 100.0
    plan = FramePlan(frame_time, 360.0)
    live = float(plan.live_time_s)
    image_rates = eqx.tree_at(
        lambda r: (
            r.planet,
            r.star_leakage,
            r.local_zodi,
            r.exozodi,
            r.n_pix,
            r.variance.planet,
            r.variance.star_leakage,
            r.variance.local_zodi,
            r.variance.exozodi,
            r.variance.n_pix,
        ),
        electrons,
        (*image[0], pixels[0], *image[1], pixels[1]),
    )
    _, variance = detector.mean_and_variance(image_rates, plan)
    residual = (pp.speckle_residual_fraction * image[0, 1] * live) ** 2 + (
        pp.exozodi_residual_fraction * image[0, 3] * live
    ) ** 2
    image_snr = image[0, 0] * live / np.sqrt(variance + residual)
    kwargs = dict(frame_time_s=frame_time, post_processing=pp)
    snr = ox.snr(c["rates"], detector, live, **kwargs)
    solved = ox.exposure_time(c["rates"], detector, image_snr, **kwargs)

    dp = abs(c["error"][0, 0])
    db = np.maximum(abs(c["error"][1]).max(axis=0), abs(c["pixel_ratio"][1] - 1)) + abs(
        c["error"][0, 3]
    )
    snr_band, time_band = 2 * (dp + db / 2), 2 * (2 * dp + db)
    assert np.all(snr_band < 3e-2) and np.all(time_band < 6e-2)
    assert np.all(snr > 1.0) and np.all(image_snr > 1.0)
    assert np.all(abs(snr / image_snr - 1) <= snr_band)
    assert np.all(solved.valid)
    assert np.all(abs(solved.time_s / live - 1) <= time_band)
