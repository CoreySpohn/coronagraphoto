"""Read and build cost of the aperture record against the planner budget."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from conftest import make_path, make_system, with_zodi, write_yip
from optixstuff import Detector
from optixstuff.spectral import Band
from skyscapes import LeinertZodi
from yippy import open_yip

import coronagraphoto as cp


@pytest.mark.medium(
    reason="YIP load, one response and 180 timed batches of 40000 reads"
)
def test_record_meets_the_planner_budget(tmp_path):
    """Fails if a read, a per-target build or one exposure outgrows its budget.

    A yield planner builds the record once per target and reads it once per
    drawn planet per decision, each decision at its own time, so the read
    with its time sets the speed. The local zodi is the pointing-dependent
    Leinert model, whose radiance the read evaluates at the eight nodes.
    Budget and measured cost on a 96 by 96 pixel detector, two bands of four
    nodes, one Apple M-series core:

    - Read under ``jit(vmap(vmap(...)))`` over 4000 draws of 10 planets at
      one shared time, each returning S/N at 10 h and the time to S/N 7 in
      both bands: budget about 1 microsecond per draw-planet, measured
      0.26. Fails above 5.
    - The same 40000 reads as 16 times of 250 draws of 10 planets, the time
      batched on the outer axis: measured 0.25 per draw-planet. Fails
      above 5.
    - The same 40000 reads with a different time on every read, so every
      read pays its own radiance: measured 0.24. Fails above 5.
    - Per-target build, ``aperture_record`` after compilation: budget 10 s,
      measured 0.08 to 0.10 s (0.3 to 1.3 s on the first call, with
      compilation). Fails above 10 s.
    - One exposure of a known planet, a single compiled read with its S/N:
      budget 0.1 s, measured 2e-5 s. Fails above 0.1 s.
    - The instrument response is outside the per-target budget: measured
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
    scene = with_zodi(make_system(separation_arcsec=2.3 * scale), LeinertZodi())
    bands = (Band(500.0, 550.0), Band(550.0, 600.0))
    exposure = cp.build_exposure(scene, instrument, bands, time_d=0.0)

    start = time.perf_counter()
    response = cp.aperture_response(
        instrument, exposure.quadrature, n_separations=33, n_angles=16
    )
    response_s = time.perf_counter() - start

    def build():
        start = time.perf_counter()
        record = cp.aperture_record(scene, instrument, exposure, response)
        jax.block_until_ready(record)
        return record, time.perf_counter() - start

    record, first_build_s = build()
    build_s = np.median([build()[1] for _ in range(5)])

    pp = ox.PostProcessing(1e-4, 0.01, background_variance_factor=1.0)

    def read(east, north, contrast, time_d):
        rates = cp.aperture_rates(record, east, north, contrast=contrast, time_d=time_d)
        kwargs = dict(frame_time_s=100.0, post_processing=pp)
        return (
            ox.snr(rates, detector, 36000.0, **kwargs),
            ox.exposure_time(rates, detector, 7.0, **kwargs).time_s,
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
    # The synthetic table has 33 rings but its source nodes end inside the
    # last ones: draw within the rings the coronagraph supports.
    rings = np.flatnonzero(np.asarray(response.throughput_supported).all(axis=(0, 1)))
    last = float(response.separation_step_arcsec) * int(rings.max())
    separations = jnp.asarray(rng.uniform(0.05 * last, 0.95 * last, shape))
    angles = jnp.deg2rad(jnp.asarray(rng.uniform(0.0, 360.0, shape)))
    easts, norths = separations * jnp.sin(angles), separations * jnp.cos(angles)
    contrasts = jnp.asarray(10 ** rng.uniform(-10.5, -9.0, shape))
    planets = (easts, norths, contrasts)
    shared = (0, 0, 0, None)
    batched = jax.jit(jax.vmap(jax.vmap(read, shared), shared))
    read_us = 1e6 * timed(batched, *planets, 200.0, repeats=60) / separations.size
    # A decision per time: 16 times of 250 draws of 10 planets.
    by_time = jax.jit(jax.vmap(jax.vmap(jax.vmap(read, shared), shared)))
    decisions = tuple(x.reshape(16, 250, 10) for x in planets)
    times = jnp.asarray(rng.uniform(0.0, 3650.0, 16))
    times_us = 1e6 * timed(by_time, *decisions, times, repeats=60) / separations.size
    # The worst case: no two reads share a time.
    each = jax.jit(jax.vmap(jax.vmap(read)))
    every = jnp.asarray(rng.uniform(0.0, 3650.0, shape))
    each_us = 1e6 * timed(each, *planets, every, repeats=60) / separations.size
    exposure_s = timed(jax.jit(read), 0.025, 0.043, 1e-10, 200.0, repeats=20)
    print(
        f"aperture budget: read {read_us:.3f} us per draw-planet at one time, "
        f"{times_us:.3f} us with 16 times batched, {each_us:.3f} us with a time "
        f"per read, build {build_s:.3f} s (first {first_build_s:.2f} s), "
        f"one exposure {exposure_s:.2e} s, response {response_s:.1f} s"
    )
    # The timed reads are real: every draw lies inside the supported tables, and
    # the local zodi they scale is neither absent nor the same at every time.
    assert np.all(np.isfinite(batched(*planets, 200.0)[0]))
    assert np.all(np.isfinite(each(*planets, every)[0]))
    zodi = jax.vmap(
        lambda t: (
            cp.aperture_rates(record, 0.025, 0.043, contrast=1e-10, time_d=t).local_zodi
        )
    )(times)
    assert np.all(zodi > 0) and float(zodi.max() / zodi.min()) > 1.5
    assert read_us < 5.0
    assert times_us < 5.0
    assert each_us < 5.0
    assert build_s < 10.0
    assert exposure_s < 0.1
