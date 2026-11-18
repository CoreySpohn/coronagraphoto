"""Regression guards for compiled reader arguments, reuse and bounded storage."""

import gc
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_path, make_system
from hwoutils.spectral import Band, Quadrature
from optixstuff import Exposure, FramePlan
from skyscapes.scene import FlatStar

from coronagraphoto import clear_reader_cache, expected, observe, photon_rates


@pytest.mark.medium(reason="observation compiled in both argument forms")
def test_path_argument_matches_closure():
    """Fails if detector geometry traces; observation must match closure bitwise."""
    system = make_system(star=FlatStar(jnp.asarray(1e6)))
    path = make_path(n=8, step=0.5)
    path = eqx.tree_at(
        lambda p: (p.detector.read_noise_e_rms, p.detector.excess_noise_factor_sq),
        path,
        (jnp.asarray(1.3), jnp.asarray(1.7)),
    )
    bands = (Band(540.0, 560.0),)
    # Deliberately unequal weights and nondefault nodes exercise the static key.
    quadrature = Quadrature(
        bands, jnp.array([[543.0, 557.0]]), jnp.array([[7.0, 13.0]])
    )
    plan = FramePlan(10.0, 1.0)

    def read(s, p, key, t):
        exposure = Exposure(time_d=t, roll_deg=17, quadrature=quadrature, plan=plan)
        return observe(key, s, p, exposure)

    key, time_d = jax.random.key(0), jnp.asarray(0.0)
    argument = eqx.filter_jit(read)(system, path, key, time_d)
    closure = eqx.filter_jit(lambda s, k, t: read(s, path, k, t))(system, key, time_d)
    for a, b in zip(jax.tree.leaves(argument), jax.tree.leaves(closure), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    with pytest.raises(ValueError, match="quadrature must be concrete host geometry"):
        eqx.filter_jit(lambda e: observe(key, system, path, e))(
            Exposure(time_d=time_d, roll_deg=17, quadrature=quadrature, plan=plan)
        )


@pytest.mark.parametrize(
    "reader",
    [
        photon_rates,
        expected,
        pytest.param(
            observe,
            marks=pytest.mark.medium(
                reason="two cold observation kernels test cache hit and invalidation"
            ),
        ),
    ],
)
def test_eager_kernel_reuse(reader, caplog):
    """Fails on eager recompilation after time/flux changes; compile-log delta=0.

    The same concrete quadrature must share a kernel after time/flux changes.
    A different quadrature object must compile and its weights affect output.
    observe must reuse the expected renderer.
    """
    clear_reader_cache()
    system = make_system(star=FlatStar(jnp.asarray(1e6)))
    varied = eqx.tree_at(lambda s: s.star.flux_phot_per_nm_m2, system, jnp.asarray(2e6))
    path = make_path(n=8, step=0.5, chromatic=True)
    bands = (Band(535.0, 565.0),)
    nodes, weights = np.array([[550.0]]), np.array([[30.0]])
    quadrature = Quadrature(bands, nodes, weights)
    changed = Quadrature(bands, nodes + 1, weights * 2)
    key, plan = jax.random.key(0), FramePlan(1e6, 1.0)

    def args(system, rule, time_d):
        exposure = Exposure(time_d=time_d, roll_deg=23, quadrature=rule, plan=plan)
        if reader is observe:
            return key, system, path, exposure
        return system, path, exposure

    # Build all inputs before capturing logs: the guard concerns reader calls.
    t0, t1 = jnp.asarray(0.0), jnp.asarray(1.0)
    with jax.log_compiles(True), caplog.at_level("WARNING"):
        first = jax.block_until_ready(reader(*args(system, quadrature, t0)))
        if reader is photon_rates:
            assert first.quadrature is quadrature
        assert any("Compiling jit(" in r.message for r in caplog.records)
        caplog.clear()
        second = jax.block_until_ready(reader(*args(varied, quadrature, t1)))
        assert not any("Compiling jit(" in r.message for r in caplog.records)
        if reader is observe:
            caplog.clear()
            jax.block_until_ready(expected(*args(varied, quadrature, t1)[1:]))
            assert not any("Compiling jit(" in r.message for r in caplog.records)
        caplog.clear()
        different = jax.block_until_ready(reader(*args(varied, changed, t1)))
        assert any("Compiling jit(" in r.message for r in caplog.records)
    field = (lambda x: x.data) if reader is observe else (lambda x: x.star_leakage)
    assert np.any(np.asarray(field(first)) != np.asarray(field(second)))
    assert np.any(np.asarray(field(second)) != np.asarray(field(different)))


def test_cache_eviction_is_bitwise(caplog, monkeypatch):
    """Fails if configurations accumulate or eviction changes reader bits.

    Five pointings exceed the four-entry eager cache. Returning to the first
    must compile again and reproduce every output bit, including signed zero.
    """
    from coronagraphoto.simulation import _cached_image_kernel

    clear_reader_cache()
    kernels = []
    jit = jax.jit

    def track_kernel(fun):
        kernel = jit(fun)
        kernels.append(weakref.ref(kernel))
        return kernel

    monkeypatch.setattr(jax, "jit", track_kernel)
    system = make_system(star=FlatStar(jnp.asarray(1e6)))
    path = make_path(n=8, step=0.5)
    bands = (Band(540.0, 560.0),)
    q = Quadrature.build(bands, system, path, order=1)
    scene = system

    def at_roll(angle):
        return photon_rates(
            scene, path, Exposure(time_d=0.0, roll_deg=angle, quadrature=q)
        )

    outputs = [jax.block_until_ready(at_roll(angle)) for angle in range(5)]
    assert _cached_image_kernel.cache_info().currsize == 4
    gc.collect()
    assert kernels[0]() is None
    with jax.log_compiles(True), caplog.at_level("WARNING"):
        caplog.clear()
        restored = jax.block_until_ready(at_roll(0))
        assert any("Compiling jit(" in r.message for r in caplog.records)
    for a, b in zip(
        jax.tree.leaves(outputs[0]), jax.tree.leaves(restored), strict=True
    ):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    clear_reader_cache()
    assert _cached_image_kernel.cache_info().currsize == 0
    gc.collect()
    assert all(kernel() is None for kernel in kernels)
