"""Read and build cost of the aperture record against the planner budget."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_path, make_system, with_zodi, write_yip
from optixstuff import Detector
from optixstuff.spectral import Band
from yippy import open_yip
from zodi.jax import PrecomputedZodi

import coronagraphoto as cp


@pytest.mark.medium(
    reason="YIP load, one calibration and 60 timed batches of 40000 reads"
)
def test_record_meets_the_planner_budget(tmp_path):
    """Fails if a read, a per-target build or one exposure outgrows its budget.

    A yield planner reads the record once per drawn planet per decision, so
    the read sets its speed. Budget and measured cost on a 96 by 96 pixel
    detector, two bands of four nodes, one Apple M-series core:

    - Read under ``jit(vmap(vmap(...)))`` over 4000 draws of 10 planets, each
      returning S/N at 10 h and the time to S/N 7 in both bands: budget about
      1 microsecond per draw-planet, measured 0.22. Fails above 5.
    - Per-target build, ``aperture_record`` after compilation: budget 10 s,
      measured 0.08 to 0.10 s (0.3 to 1.3 s on the first call, with
      compilation). Fails above 10 s.
    - One exposure of a known planet, a single compiled read with its S/N:
      budget 0.1 s, measured 2e-5 s. Fails above 0.1 s.
    - The instrument calibration is outside the per-target budget: measured
      3.0 s here for 33 rings of 16 sources.

    Thresholds are wall-clock on a shared machine, hence loose; the medians
    are printed for the record.
    """
    coro = open_yip(write_yip(tmp_path / "yip")).prepare()
    base = eqx.tree_at(lambda p: p.coronagraph, make_path(), coro)
    scale = float(base.primary.frame(550.0).lod_to_arcsec(550.0))
    detector = Detector(
        shape=(96, 96),
        pixel_scale_arcsec=0.1 * scale,
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=3e-4,
        clock_induced_charge_e_per_pixel_frame=0.005,
        read_noise_e_rms=0.2,
    )
    instrument = eqx.tree_at(lambda p: p.detector, base, detector)
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    scene = with_zodi(make_system(separation_arcsec=2.3 * scale), zodi)
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    exposure = cp.build_exposure(scene, instrument, bands, time_d=0.0)

    start = time.perf_counter()
    calibration = cp.aperture_calibration(
        instrument, exposure.quadrature, n_separations=33, n_angles=16
    )
    calibration_s = time.perf_counter() - start

    def build():
        start = time.perf_counter()
        record = cp.aperture_record(scene, instrument, exposure, calibration)
        jax.block_until_ready(record)
        return record, time.perf_counter() - start

    record, first_build_s = build()
    build_s = np.median([build()[1] for _ in range(5)])

    pp = cp.PostProcessing(1e-4, 0.01)

    def read(separation, angle, contrast):
        rates = cp.aperture_rates(
            record, separation, contrast, position_angle_deg=angle
        )
        kwargs = dict(frame_time_s=100.0, post_processing=pp)
        return (
            cp.snr(rates, detector, 36000.0, **kwargs),
            cp.exposure_time(7.0, rates, detector, **kwargs).time_s,
        )

    def timed(function, *args, repeats):
        jax.block_until_ready(function(*args))
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(function(*args))
            times.append(time.perf_counter() - start)
        return np.median(times)

    rng = np.random.default_rng(0)
    shape = (4000, 10)
    last = float(calibration.separation_step_arcsec) * 32
    separations = jnp.asarray(rng.uniform(0.05 * last, 0.95 * last, shape))
    angles = jnp.asarray(rng.uniform(0.0, 360.0, shape))
    contrasts = jnp.asarray(10 ** rng.uniform(-10.5, -9.0, shape))
    batched = jax.jit(jax.vmap(jax.vmap(read)))
    batch_s = timed(batched, separations, angles, contrasts, repeats=60)
    read_us = 1e6 * batch_s / separations.size
    exposure_s = timed(jax.jit(read), 0.05, 30.0, 1e-10, repeats=20)
    print(
        f"aperture budget: read {read_us:.3f} us per draw-planet, "
        f"build {build_s:.3f} s (first {first_build_s:.2f} s), "
        f"one exposure {exposure_s:.2e} s, calibration {calibration_s:.1f} s"
    )
    # The timed reads are real: every draw lies inside the tables.
    assert np.all(np.isfinite(batched(separations, angles, contrasts)[0]))
    assert read_us < 5.0
    assert build_s < 10.0
    assert exposure_s < 0.1
