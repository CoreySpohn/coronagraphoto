"""Integral count-rate and fixed-time S/N closure of two photon-budget readers."""

from dataclasses import replace

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import jaxedith as je
import numpy as np
import pytest
from conftest import make_path, make_system, write_yip
from hwoutils.constants import arcsec2rad
from hwoutils.grids import Image, PixelGrid, SkyOrientation
from hwoutils.spectral import Band, Quadrature
from optixstuff import Detector, ElectronRates, FramePlan, YippyCoronagraph
from skyscapes.background import PrecomputedZodi
from skyscapes.disk import GraterDisk
from yippy import Circular
from yippy.performance import _performance_curves
from zodi import SurfaceBrightnessDust

from coronagraphoto import image_electrons

TERMS = ("planet", "star_leakage", "local_zodi", "exozodi")


def _richardson(values):
    """Three halving levels: signed extrapolation and GCI with F_s=1.25.

    Non-monotone or non-shrinking ladders have no defensible GCI band;
    return NaN so callers must reject them before claiming closure.
    """
    first, second = np.diff(values, axis=0)
    order = np.log2(abs(first / second))
    valid = (first * second > 0) & (order > 0)
    correction = second / (2**order - 1)
    limit = np.where(valid, values[2] + correction, np.nan)
    uncertainty = np.where(valid, 1.25 * abs(correction), np.nan)
    return limit, order, uncertainty


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    """Render one physical scene at three pitches; cache every measured SRQ.

    Four image structures are necessary: three closure pitches and p/8
    to measure the next-finer gap for p/4 and the four-level disk spread.
    Detector shape is static.
    The native YIP is a Gaussian, sigma=0.7 lod, integral=0.3, pitch=0.5
    lod, with nonzero curved wings. No response is mocked or renormalized.
    """
    rule = Circular(0.7)
    coro = YippyCoronagraph(
        write_yip(tmp_path_factory.mktemp("calculator-closure") / "yip"),
        ensure_psf_datacube=True,
        rule=rule,
    )
    base = eqx.tree_at(lambda p: p.coronagraph, make_path(), coro)
    scale = float(base.primary.frame(550.0).lod_to_arcsec(550.0))
    disk = GraterDisk(
        sma_AU=1.0,
        alpha_in=2.0,
        alpha_out=-2.0,
        ksi0_AU=0.05,
        gamma=2.0,
        beta=1.0,
        rmin_AU=0.1,
        rmax_AU=5.0,
        wavelengths_nm=jnp.array([500.0, 600.0]),
        g_HG_grid=jnp.zeros(2),
        Ag_grid=jnp.ones(2),
        amplitude=SurfaceBrightnessDust(level=3.0),
        n_slices_los=24,
    )
    system = make_system(separation_arcsec=2.3 * scale, sky_pa_deg=270.0, disk=disk)
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    pointing, time_d = SkyOrientation(0.0), 0.0
    q = je.aperture_quadrature(system, base, bands, zodi=zodi)
    image_q = Quadrature.build(bands, None, 4)
    prepared = je.prepare_aperture(base, bands, q, rule=rule)
    # Both readers use this host-prepared rule and exactly these spectral nodes.
    np.testing.assert_array_equal(q.nodes_nm, image_q.nodes_nm)
    np.testing.assert_array_equal(q.weights_nm, image_q.weights_nm)
    np.testing.assert_array_equal(q.nodes_nm, prepared.quadrature.nodes_nm)
    np.testing.assert_array_equal(q.weights_nm, prepared.quadrature.weights_nm)
    rates = je.aperture_rates(
        system,
        base,
        bands,
        time_d=time_d,
        planet_index=0,
        rule=rule,
        quadrature=q,
        prepared_aperture=prepared,
        zodi=zodi,
    )
    calculator = base.detector.electrons(rates, q)
    expected = np.stack([np.asarray(getattr(calculator, name)) for name in TERMS])
    east, north = system.positions(jnp.array([time_d]))[:2, 0, 0]
    x, y = pointing.sky_to_detector(east, north)
    # Reevaluate the record's subpixel quadrature at the actual node positions.
    # This isolates its resolution error from interpolation of tabulated curves.
    calc_ladder, analytic_nodes = [], []
    for oversample in (10, 20, 40):
        node_rates = []
        for waves, nodes in zip(np.asarray(q.nodes_nm), prepared.nodes, strict=True):
            row, analytic_row = [], []
            for wave, (_, node_rule, record, angular_scale) in zip(
                waves, nodes, strict=True
            ):
                separation = float(jnp.hypot(x, y) / angular_scale)

                def offaxis(s, wave=wave):
                    image = coro.source_psfs(
                        wave, jnp.array([s]), jnp.zeros(1), grid=coro.grid
                    )
                    return replace(image, data=image.data[0], leading_axes=())

                refined = _performance_curves(
                    offaxis,
                    coro.stellar_intens(0.0),
                    record.rule,
                    separations_lod=[separation],
                    oversample=oversample,
                )
                phi = system.planet_flux_densities(wave, jnp.array([time_d]))[0, 0]
                gain = base.primary.collecting_area_m2 * base.system_throughput(wave)
                row.append(gain * phi * refined.throughput[0])
                analytic_row.append(
                    gain
                    * phi
                    * 0.3
                    * (1 - np.exp(-(node_rule.radius_lod**2) / (2 * 0.7**2)))
                )
            node_rates.append(jnp.stack(row))
            if oversample == 10:
                analytic_nodes.append(jnp.stack(analytic_row))
        calc_ladder.append(
            np.asarray(
                base.detector.electrons(
                    replace(rates, planet=jnp.stack(node_rates)), q
                ).planet
            )
        )
    calc_ladder = np.asarray(calc_ladder)
    analytic = np.asarray(
        base.detector.electrons(
            replace(rates, planet=jnp.stack(analytic_nodes)), q
        ).planet
    )
    calc_limit, calc_order, calc_uncertainty = _richardson(calc_ladder)
    calc_discretization = expected[0] - calc_limit
    centre_grid = PixelGrid.regular(
        (1, 1),
        1.0,
        "arcsec",
        center=(-float(east), -float(north)),
        orientation=SkyOrientation(0.0, parity=1),
    )
    centre = np.asarray(
        system.photon_radiance(q.nodes_nm, time_d, grid=centre_grid).data
    )
    centre = centre[..., 0, 0]
    measured, bounds, redistribution, snrs, snr_bounds = [], [], [], [], []
    dust_ratios = []
    plan = FramePlan(100.0, 36.0)
    pp = je.PostProcessing(0.01, 0.03)
    for n, pitch in ((48, 0.2), (96, 0.1), (192, 0.05), (384, 0.025)):
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
        images = image_electrons(
            system,
            path,
            bands,
            time_d=time_d,
            pointing=pointing,
            quadrature=prepared.quadrature,
            zodi=zodi,
        )
        sky = PixelGrid.regular((n, n), pitch * scale, "arcsec", orientation=pointing)
        weights = []
        for band_index, (_, ref, angular_scale, record) in enumerate(
            prepared.references
        ):
            frame = path.primary.frame(ref, record.frame.diameter_convention)
            grid = PixelGrid.regular(
                (n, n), pitch * scale / float(angular_scale), "lod", frame=frame
            )
            psf = Image(
                images.planet[band_index],
                grid,
                (),
                "per_pixel",
                "stellar_flux_fraction",
            )
            weights.append(
                record.rule.weights(psf, (x / angular_scale, y / angular_scale))
            )
        weights = jnp.stack(weights)
        sums = jax.tree.map(lambda a, w=weights: jnp.sum(a * w, axis=(-2, -1)), images)
        measured.append(np.stack([np.asarray(getattr(sums, name)) for name in TERMS]))
        # Model-form difference a: aperture-average raw radiance versus centre.
        radiance = np.asarray(system.photon_radiance(q.nodes_nm, time_d, grid=sky).data)
        average = np.sum(radiance * np.asarray(weights)[:, None], axis=(-2, -1))
        average /= np.asarray(weights.sum(axis=(-2, -1)))[:, None]
        disk_difference = replace(
            rates, exozodi=rates.exozodi * jnp.asarray(average / centre - 1)
        )
        bounds.append(np.asarray(detector.electrons(disk_difference, q).exozodi))
        # Model-form difference b includes native deposition and PSF redistribution.
        # Compute the unconvolved aperture flux independently from sky radiance.
        raw_nodes = []
        for row, waves in enumerate(np.asarray(q.nodes_nm)):
            raw_row = []
            for node, wave in enumerate(waves):
                transmission = coro.background_transmission(wave, grid=coro.grid).data
                # This fixture has spatially constant background transmission.
                np.testing.assert_allclose(transmission, 0.3, rtol=1e-14)
                raw_row.append(
                    base.primary.collecting_area_m2
                    * base.system_throughput(wave)
                    * np.sum(radiance[row, node] * np.asarray(weights[row]))
                    * sky.pixel_area
                    * arcsec2rad**2
                    * transmission[0, 0]
                )
            raw_nodes.append(jnp.stack(raw_row))
        raw = detector.electrons(
            replace(rates, exozodi=jnp.stack(raw_nodes)), q
        ).exozodi
        redistribution.append(np.asarray(sums.exozodi - raw))
        if n == 48:
            doubled = eqx.tree_at(lambda s: s.disk.amplitude.level, system, 6.0)
            doubled_rates = je.aperture_rates(
                doubled,
                path,
                bands,
                time_d=time_d,
                planet_index=0,
                rule=rule,
                quadrature=q,
                prepared_aperture=prepared,
                zodi=zodi,
            )
            doubled_images = image_electrons(
                doubled,
                path,
                bands,
                time_d=time_d,
                pointing=pointing,
                quadrature=q,
                zodi=zodi,
            )
            dust_ratios = np.stack(
                (
                    detector.electrons(doubled_rates, q).exozodi / calculator.exozodi,
                    jnp.sum(doubled_images.exozodi * weights, axis=(-2, -1))
                    / sums.exozodi,
                )
            )
        n_pix = weights.sum(axis=(-2, -1))
        _, variance = detector.mean_and_variance(sums, plan, n_pix=n_pix)
        variance += (
            pp.speckle_residual_fraction * sums.star_leakage * plan.live_time_s
        ) ** 2
        variance += (
            pp.exozodi_residual_fraction * sums.exozodi * plan.live_time_s
        ) ** 2
        image_snr = sums.planet * plan.live_time_s / jnp.sqrt(variance)
        calc_snr = je.snr(rates, detector, q, plan, n_pix=n_pix, post_processing=pp)
        snrs.append(np.stack((calc_snr, image_snr)))
        snr_bounds.append((detector, n_pix))
    measured, bounds, snrs = np.asarray(measured), np.asarray(bounds), np.asarray(snrs)
    redistribution = np.asarray(redistribution)
    gaps = abs(np.diff(measured, axis=0))
    disk_ladder = measured[:, 3].copy()
    disk_spread = np.max(disk_ladder, axis=0) - np.min(disk_ladder, axis=0)
    measured, bounds, redistribution, snrs = (
        a[:3] for a in (measured, bounds, redistribution, snrs)
    )
    image_limit, image_order, image_uncertainty = _richardson(measured[:, 0])
    planet_tolerance = image_uncertainty + calc_uncertainty + 1e-9 * abs(expected[0])
    tolerance = gaps / 2 + 1e-9 * abs(expected)
    all_tolerances = tolerance.copy()
    # Transport the planet limit bands to each production resolution by the
    # measured distance to its limit. Add intervals, never root-sum-square them.
    all_tolerances[:, 0] = (
        abs(measured[:, 0] - image_limit) + planet_tolerance + abs(calc_discretization)
    )
    all_tolerances[:, 3] = (
        disk_spread + 1e-9 * abs(expected[3]) + abs(bounds) + abs(redistribution)
    )
    snr_tolerance, snr_intervals = [], []
    for index, (detector, n_pix) in enumerate(snr_bounds[:3]):
        # Interval propagation, without cancellation or quadrature of errors.
        low, high = [], []
        for sign, output in ((-1, low), (1, high)):
            values = {}
            for term_index, name in enumerate(TERMS):
                values[name] = jnp.maximum(
                    getattr(calculator, name)
                    + sign * all_tolerances[index, term_index],
                    0,
                )
            interval = ElectronRates(
                **values, thermal=jnp.zeros(2), stellar_companions=jnp.zeros(2)
            )
            _, variance = detector.mean_and_variance(interval, plan, n_pix=n_pix)
            variance += (
                pp.speckle_residual_fraction * interval.star_leakage * plan.live_time_s
            ) ** 2
            variance += (
                pp.exozodi_residual_fraction * interval.exozodi * plan.live_time_s
            ) ** 2
            output.extend(
                (np.asarray(interval.planet * plan.live_time_s), np.asarray(variance))
            )
        lower, upper = low[0] / np.sqrt(high[1]), high[0] / np.sqrt(low[1])
        snr_intervals.append(np.stack((lower, upper)))
        snr_tolerance.append(np.maximum(snrs[index, 0] - lower, upper - snrs[index, 0]))
    print("\ncalculator e/s", expected.tolist())
    print("image e/s [sampling,term,band]", measured.tolist())
    print("gaps", gaps.tolist(), "tolerance", tolerance.tolist())
    print("disk a", bounds.tolist(), "disk b", redistribution.tolist())
    print("dust ratios", dust_ratios.tolist())
    print("planet orders", np.log2(gaps[0, 0] / gaps[1, 0]).tolist())
    print("S/N [sampling,reader,band]", snrs.tolist())
    print("S/N tolerances", np.asarray(snr_tolerance).tolist())
    evidence = dict(
        calculator=expected,
        images=measured,
        gaps=gaps,
        tolerance=all_tolerances,
        disk_a=bounds,
        disk_b=redistribution,
        snr=snrs,
        snr_tolerance=np.asarray(snr_tolerance),
        snr_intervals=np.asarray(snr_intervals),
        dust_ratios=dust_ratios,
        planet_image_limit=image_limit,
        planet_image_order=image_order,
        planet_image_uncertainty=image_uncertainty,
        planet_calc_ladder=calc_ladder,
        planet_calc_limit=calc_limit,
        planet_calc_order=calc_order,
        planet_calc_uncertainty=calc_uncertainty,
        planet_tolerance=planet_tolerance,
        planet_calc_discretization=calc_discretization,
        planet_analytic=analytic,
        planet_image_analytic_distance=abs(image_limit - analytic),
        planet_calc_analytic_distance=abs(calc_limit - analytic),
        disk_ladder=disk_ladder,
        disk_spread=disk_spread,
    )
    print("closure evidence", {key: value.tolist() for key, value in evidence.items()})
    return evidence


@vv.case(
    "etc-count-rates",
    "code-verification",
    srq="planet count rate Cp; stellar leakage CRbs",
)
@vv.seam("jaxedith", "coronagraphoto")
def test_planet_and_leakage_closure(closure):
    """Fails if the planet limits or reflected stellar aperture disagree.

    Smooth pixel-coverage quadrature is second order: measured orders in
    1.7--2.3 include the sub-asymptotic blue image and calculator ladders.
    Both limits use signed Richardson extrapolation and GCI F_s=1.25;
    their absolute uncertainties plus 1e-9*abs(C) form the closure band.
    D_calc is the measured signed default-record discretization error.
    Leakage retains half the next-pitch gap plus the 1e-9 relative floor.
    """
    c = closure
    for order in (c["planet_image_order"], c["planet_calc_order"]):
        assert np.all((order > 1.7) & (order < 2.3))
    assert np.all(
        abs(c["planet_image_limit"] - c["planet_calc_limit"]) <= c["planet_tolerance"]
    )
    assert np.all(
        abs(c["calculator"][0] - c["planet_calc_limit"])
        <= abs(c["planet_calc_discretization"]) + c["planet_calc_uncertainty"]
    )
    assert np.all(abs(c["images"][:, 1] - c["calculator"][1]) <= c["tolerance"][:, 1])


@vv.case("etc-count-rates", "code-verification", srq="planet count rate Cp")
@vv.seam("jaxedith", "coronagraphoto")
def test_survival_applied_once(closure):
    """Fails on duplicate survival: the mutant is 0.9*C, 10% low.

    Compare extrapolated planet rates with the same GCI closure band;
    both chromatic QE and survival=0.9 remain on the real detector paths.
    The production-default mutant must also lie outside that band.
    """
    c = closure
    image, calc, tolerance = (
        c["planet_image_limit"],
        c["planet_calc_limit"],
        c["planet_tolerance"],
    )
    assert np.all(abs(image / calc - 1) <= tolerance / abs(calc))
    assert np.all(abs(image - 0.9 * calc) > tolerance)
    assert np.all(abs(image - 0.9 * c["calculator"][0]) > tolerance)


@vv.case("etc-count-rates", "code-verification", srq="zodi and exozodi rates")
@vv.seam("jaxedith", "coronagraphoto")
def test_background_closure(closure):
    """Fails on omitted disk, wrong solid angle or unconverged disk sums.

    Local zodi retains the gap/2 plus 1e-9 relative floor. Exozodi has no
    claimed convergence order: sky deposition onto the fixed native PSF
    grid reverses sign along the pitch ladder. S_disk is the empirical
    spread (max minus min) over p, p/2, p/4, p/8. Measured sums in e/s:
    blue=(0.0241267824200, 0.0241288511279, 0.0241230525276,
          0.0241225491126),
    red=(0.0257179187191, 0.0257180986553, 0.0257129038615,
         0.0257130461716).
    The convergence-only check S_disk < 1e-3*abs(C) catches a divergent
    ladder; its residual is an epistemic band, not a Richardson estimate.
    At every closure pitch use S_disk + abs(a) + abs(b) + 1e-9*abs(C).
    Measured model-form differences a (raw aperture-average versus center)
    and b (convolved versus raw sum times transmission) compare different
    disk models, not numerical errors. Dust doubling is linear to a 1e-12
    arithmetic allowance; both readers must give nonzero exozodi.
    """
    c = closure
    np.testing.assert_allclose(c["dust_ratios"], 2.0, rtol=1e-12)
    assert np.all(c["calculator"][3] > 0) and np.all(c["images"][:, 3] > 0)
    assert np.all(abs(c["images"][:, 2] - c["calculator"][2]) <= c["tolerance"][:, 2])
    assert np.all(c["disk_spread"] < 1e-3 * abs(c["calculator"][3]))
    assert np.all(abs(c["images"][:, 3] - c["calculator"][3]) <= c["tolerance"][:, 3])


@vv.case("etc-inversion", "code-verification", srq="S/N at fixed time")
@vv.seam("jaxedith", "coronagraphoto")
def test_snr_closure(closure):
    """Fails if the image budget and calculator give inconsistent S/N.

    Basis: interval propagation of all four term bounds through detector
    mean_and_variance and residual count squares at each closure pitch.
    Exozodi contributes its empirical spread and measured model-form
    differences, with no order claim. Effective pixel count is sum of
    fractional aperture areas, shared by both aperture budgets.
    """
    snrs = closure["snr"]
    assert np.all(abs(snrs[:, 0] - snrs[:, 1]) <= closure["snr_tolerance"])
