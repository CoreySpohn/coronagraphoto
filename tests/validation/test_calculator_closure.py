"""Closure of the aperture record, S/N and exposure time against image sums."""

import equinox as eqx
import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from conftest import closure_system, make_path, write_yip
from optixstuff import Detector, FramePlan
from optixstuff.apertures import Circular
from optixstuff.grids import Image, Measure, SkyOrientation
from optixstuff.spectral import Band
from yippy import open_yip

import coronagraphoto as cp

TERMS = ("planet", "star_leakage", "local_zodi", "exozodi")
pytestmark = pytest.mark.medium(
    reason="YIP load, scene operator and four renders at two detector pitches"
)


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
def closure(tmp_path_factory):
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
        .prepare(rules=(rule,))
        .with_scene_operator()
    )
    base = eqx.tree_at(lambda p: p.coronagraph, make_path(), coro)
    scale = float(base.primary.frame(550.0).lod_to_arcsec(550.0))
    system = closure_system(scale)
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    roll, contrast = 15.0, 1e-9
    east, north = system.positions(jnp.array([0.0]))[:2, 0, 0]
    separation = jnp.hypot(east, north)
    x, y = SkyOrientation(roll).sky_to_detector(east, north)
    pp = ox.PostProcessing(1e-4, 0.01, background_variance_factor=1.0)
    frame_time, n_frames = 100.0, 360.0
    plan = FramePlan(frame_time, n_frames)
    levels = []
    for n, pitch in ((48, 0.2), (96, 0.1)):
        detector = Detector(
            shape=(n, n),
            pixel_scale_arcsec=pitch * scale,
            qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
            survival_fraction=0.9,
            dark_current_e_per_pixel_s=3e-4,
            clock_induced_charge_e_per_pixel_frame=0.005,
            read_noise_e_rms=0.2,
        )
        path = eqx.tree_at(lambda p: p.detector, base, detector)
        exposure = cp.build_exposure(system, path, bands, time_d=0.0, roll_deg=roll)
        response = cp.aperture_response(
            path, exposure.quadrature, rule=rule, n_separations=33, n_angles=16
        )
        record = cp.aperture_record(system, path, exposure, response)
        rates = cp.aperture_rates(record, east, north, contrast=contrast, time_d=0.0)
        electrons = detector.electrons(rates)

        # The independent reader: the rule's own weights on the expected images.
        images = cp.expected(system, path, exposure)
        weights = []
        for band in bands:
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
        value = np.asarray(
            [[getattr(r, t) for t in TERMS[1:]] for r in (rates, rates.variance)]
        )
        lattice_band = (
            _bilinear_bound(
                jnp.stack([record.star_leakage, record.uniform_sky, record.exozodi]),
                row,
                column,
            ).swapaxes(0, 1)
            / value
        ).max(-1)

        def residual_sq(e, time):
            return (pp.speckle_residual_fraction * e[1] * time) ** 2 + (
                pp.exozodi_residual_fraction * e[3] * time
            ) ** 2

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
        image_snr = image[0, 0] * live / np.sqrt(variance + residual_sq(image[0], live))
        kwargs = dict(frame_time_s=frame_time, post_processing=pp)
        solved = ox.exposure_time(rates, detector, image_snr, **kwargs)

        doubled = eqx.tree_at(lambda s: s.disk.amplitude.level, system, 6.0)
        doubled_record = cp.aperture_record(doubled, path, exposure, response)
        doubled_rates = cp.aperture_rates(
            doubled_record, east, north, contrast=contrast, time_d=0.0
        )
        levels.append(
            dict(
                calculated=calculated,
                image=image,
                pixels=np.stack([electrons.n_pix, electrons.variance.n_pix]) / pixels,
                planet_band=planet_band.max(-1),
                lattice_band=lattice_band,
                snr=np.stack([ox.snr(rates, detector, live, **kwargs), image_snr]),
                time=np.asarray(solved.time_s),
                valid=np.asarray(solved.valid),
                live=live,
                dust=np.asarray(doubled_rates.exozodi / rates.exozodi),
                dust_images=np.asarray(
                    jnp.sum(cp.expected(doubled, path, exposure).exozodi * weights)
                    / jnp.sum(images.exozodi * weights)
                ),
            )
        )
    evidence = {key: np.stack([level[key] for level in levels]) for key in levels[0]}
    evidence["error"] = evidence["calculated"] / evidence["image"] - 1
    print("closure evidence", {key: value.tolist() for key, value in evidence.items()})
    return evidence


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
    planet = abs(c["error"][:, 0, 0])
    assert np.all(c["planet_band"] < 5e-3)
    assert np.all(planet <= c["planet_band"])
    # Finer pixels leave less pixel phase for the ring mean to hide.
    assert np.all(planet[1] < planet[0])
    assert np.all(abs(c["error"][:, 0, 1]) <= 1e-9)
    assert np.all(c["calculated"][:, 0, :2] > 0)


@vv.case("etc-count-rates", "code-verification", srq="planet count rate Cp")
def test_survival_applied_once(closure):
    """Fails on duplicate survival: the mutant is 0.9*C, 10% low.

    Chromatic QE and survival=0.9 stay on the real detector path of both
    readers. Basis: the planet band of the closure test; the mutant must lie
    outside it.
    """
    c = closure
    image, calculated = c["image"][:, 0, 0], c["calculated"][:, 0, 0]
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
    np.testing.assert_allclose(c["dust"], 2.0, rtol=1e-12)
    np.testing.assert_allclose(c["dust_images"], 2.0, rtol=1e-12)
    assert np.all(c["calculated"][:, 0, 2:] > 0) and np.all(c["image"][:, 0, 2:] > 0)
    assert np.all(abs(c["error"][:, 0, 2]) <= 1e-9)
    exozodi_band = 2 * c["lattice_band"][:, 0, 2] + 1e-9
    assert np.all(exozodi_band < 1e-3)
    assert np.all(abs(c["error"][:, 0, 3]) <= exozodi_band)


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
    np.testing.assert_allclose(c["pixels"][:, 0], 1.0, rtol=1e-9)
    backgrounds = abs(c["error"][:, 1, 1:]).max(axis=(-2, -1))
    pixels = abs(c["pixels"][:, 1] - 1).max(axis=-1)
    planet = abs(c["error"][:, 1, 0]).max(axis=-1)
    assert np.all(backgrounds < 1e-2) and np.all(pixels < 1e-2)
    assert np.all(planet < 3e-2)
    assert backgrounds[1] < backgrounds[0] and planet[1] < planet[0]


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
    dp = abs(c["error"][:, 0, 0])
    db = np.maximum(
        abs(c["error"][:, 1]).max(axis=-2), abs(c["pixels"][:, 1] - 1)
    ) + abs(c["error"][:, 0, 3])
    snr_band, time_band = 2 * (dp + db / 2), 2 * (2 * dp + db)
    assert np.all(snr_band < 3e-2) and np.all(time_band < 6e-2)
    assert np.all(c["snr"] > 1.0)
    assert np.all(abs(c["snr"][:, 0] / c["snr"][:, 1] - 1) <= snr_band)
    assert np.all(c["valid"])
    assert np.all(abs(c["time"] / c["live"][:, None] - 1) <= time_band)
