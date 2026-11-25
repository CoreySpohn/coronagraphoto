"""Regression guards for the traced renderer: arguments, compiles and gradients."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_path, make_system
from optixstuff import Exposure, FramePlan
from optixstuff.spectral import Band, Quadrature
from skyscapes.scene import FlatStar

from coronagraphoto import expected, observe, photon_rates


def _renders(caplog):
    """Count compiles of the shared renderer program in the captured log."""
    return sum("Compiling jit(_render)" in r.message for r in caplog.records)


def _with_nodes(quadrature, nodes):
    return eqx.tree_at(lambda q: q.nodes_nm, quadrature, nodes)


@pytest.mark.medium(reason="observation compiled in both argument forms")
def test_exposure_argument_matches_closure():
    """Fails if an exposure or detector geometry cannot be a traced argument.

    Instrument, scene and exposure as arguments must match closing over them
    bitwise, and so must a traced roll against a concrete one.
    """
    system = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    path = make_path(n=8, step=0.5)
    path = eqx.tree_at(
        lambda p: (p.detector.read_noise_e_rms, p.detector.excess_noise_factor_sq),
        path,
        (jnp.asarray(1.3), jnp.asarray(1.7)),
    )
    bands = (Band(540.0, 560.0),)
    quadrature = Quadrature(
        bands, jnp.array([[543.0, 557.0]]), jnp.array([[7.0, 13.0]])
    )
    plan = FramePlan(10.0, 1.0)
    key = jax.random.key(0)

    def read(s, p, key, t):
        exposure = Exposure(time_d=t, roll_deg=17, quadrature=quadrature, plan=plan)
        return observe(key, s, p, exposure)

    time_d = jnp.asarray(0.0)
    argument = eqx.filter_jit(read)(system, path, key, time_d)
    closure = eqx.filter_jit(lambda s, k, t: read(s, path, k, t))(system, key, time_d)
    exposure = Exposure(time_d=time_d, roll_deg=17, quadrature=quadrature, plan=plan)
    whole = eqx.filter_jit(lambda e: observe(key, system, path, e))(exposure)
    # XLA may fuse the closed-over constants differently; agree to a few ULP.
    for other in (closure, whole):
        for a, b in zip(jax.tree.leaves(argument), jax.tree.leaves(other), strict=True):
            a, b = np.asarray(a), np.asarray(b)
            if np.issubdtype(a.dtype, np.inexact):
                np.testing.assert_array_max_ulp(a, b, maxulp=4)
            else:
                np.testing.assert_array_equal(a, b)
    eager = observe(key, system, path, exposure)
    assert eager.grid.orientation.pa_y_deg == 17.0
    assert whole.grid.orientation is None


@pytest.mark.parametrize(
    "reader",
    [
        photon_rates,
        expected,
        pytest.param(
            observe,
            marks=pytest.mark.medium(reason="observation kernel compiled once"),
        ),
    ],
)
def test_reader_compiles_once_for_any_values(reader, caplog):
    """Fails on recompilation after time, flux, node, weight or roll changes.

    The compile-log count of the renderer stays at one across all of them;
    new node values and weights change the output, and observe reuses the
    program that expected compiled.
    """
    system = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    varied = eqx.tree_at(lambda s: s.star.flux_phot_per_nm_m2, system, jnp.asarray(2e6))
    path = make_path(n=8, step=0.5, chromatic=True)
    bands = (Band(535.0, 565.0),)
    quadrature = Quadrature(bands, np.array([[550.0]]), np.array([[30.0]]))
    changed = Quadrature(bands, np.array([[551.0]]), np.array([[60.0]]))
    key, plan = jax.random.key(0), FramePlan(1e6, 1.0)

    def args(system, rule, time_d, roll):
        exposure = Exposure(time_d=time_d, roll_deg=roll, quadrature=rule, plan=plan)
        if reader is observe:
            return key, system, path, exposure
        return system, path, exposure

    t0, t1 = jnp.asarray(0.0), jnp.asarray(1.0)
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level("WARNING"):
        first = jax.block_until_ready(reader(*args(system, quadrature, t0, 23)))
        assert _renders(caplog) == 1
        second = jax.block_until_ready(reader(*args(varied, quadrature, t1, 41)))
        different = jax.block_until_ready(reader(*args(varied, changed, t1, 41)))
        if reader is observe:
            jax.block_until_ready(expected(*args(varied, quadrature, t1, 41)[1:]))
        assert _renders(caplog) == 1
    if reader is photon_rates:
        assert first.quadrature is quadrature
    field = (lambda x: x.data) if reader is observe else (lambda x: x.star_leakage)
    assert np.any(np.asarray(field(first)) != np.asarray(field(second)))
    assert np.any(np.asarray(field(second)) != np.asarray(field(different)))


def test_forty_nodes_and_eight_rolls_compile_one_program(caplog):
    """Fails if a node count or a roll sequence compiles more than one program."""
    system = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    path = make_path(n=8, step=0.5, chromatic=True)
    scene = system
    band = Band(520.0, 580.0)
    nodes = np.linspace(521.0, 579.0, 40)
    jax.clear_caches()
    wide = Quadrature((band,), nodes[None], np.full((1, 40), 1.5))
    with jax.log_compiles(True), caplog.at_level("WARNING"):
        rates = [
            expected(scene, path, Exposure(0.0, roll, wide))
            for roll in np.linspace(0.0, 70.0, 8)
        ]
        jax.block_until_ready(rates)
    assert _renders(caplog) == 1
    assert not np.allclose(rates[0].planet, rates[-1].planet)


@pytest.mark.medium(reason="four reverse-mode programs and finite differences")
@pytest.mark.float64_only(reason="central finite differences at step 1e-4")
def test_roll_and_node_gradients_match_finite_differences():
    """Fails if a traced roll or node wavelength has a wrong derivative.

    The scalar mixes the planet and the leakage; central differences in
    float64 agree with jax.grad to rtol 1e-5 for the roll and each node.
    """
    system = make_system(
        star=FlatStar(jnp.asarray(1e6)), sky_pa_deg=40.0, smooth_planet=True
    )
    path = make_path(n=24, step=0.5, chromatic=True)
    scene = system
    quadrature = Quadrature(
        (Band(530.0, 570.0),),
        jnp.array([[537.0, 548.0, 561.0]]),
        jnp.array([[10.0, 20.0, 10.0]]),
    )

    def scalar(roll, nodes):
        exposure = Exposure(
            time_d=0.0, roll_deg=roll, quadrature=_with_nodes(quadrature, nodes)
        )
        rates = expected(scene, path, exposure)
        return jnp.sum(rates.planet**2) + 1e-3 * jnp.sum(rates.star_leakage**2)

    roll, nodes = 17.0, quadrature.nodes_nm
    d_roll, d_nodes = jax.grad(scalar, argnums=(0, 1))(roll, nodes)
    h = 1e-4
    fd_roll = (scalar(roll + h, nodes) - scalar(roll - h, nodes)) / (2 * h)
    np.testing.assert_allclose(d_roll, fd_roll, rtol=1e-5)
    for i in range(nodes.shape[1]):
        step = jnp.zeros_like(nodes).at[0, i].set(h)
        fd = (scalar(roll, nodes + step) - scalar(roll, nodes - step)) / (2 * h)
        np.testing.assert_allclose(d_nodes[0, i], fd, rtol=1e-5)


def test_traced_nodes_select_the_band_model_inside_the_program():
    """Fails if band selection needs concrete nodes or picks the wrong model.

    Eager calls keep one branch per model the nodes fall in; traced nodes keep
    every model and select from the node wavelength. Both give the same
    images, and a node outside every band gives NaN instead of a wrong model.
    """
    from conftest import GaussianCoronagraph
    from optixstuff import MultiBandCoronagraph
    from optixstuff.grids import Frame

    blue = GaussianCoronagraph(n=16, bandpass=Band(490, 510))
    red = GaussianCoronagraph(
        n=20, step=0.35, frame=Frame(600.0, 3.0, "inscribed"), bandpass=Band(590, 610)
    )
    path = make_path(n=16, step=0.5)
    path = eqx.tree_at(lambda p: p.coronagraph, path, MultiBandCoronagraph((blue, red)))
    scene = make_system(star=FlatStar(jnp.asarray(1e6)), smooth_planet=True)
    quadrature = Quadrature(
        (Band(490.0, 610.0),),
        jnp.array([[495.0, 600.0, 505.0]]),
        jnp.array([[1.0, 1.0, 1.0]]),
    )
    exposure = Exposure(time_d=0.0, roll_deg=10.0, quadrature=quadrature)
    eager = photon_rates(scene, path, exposure)
    traced = eqx.filter_jit(lambda e: photon_rates(scene, path, e))(exposure)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(traced), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    assert not np.allclose(eager.planet[0, 0], eager.planet[0, 1])
    outside = eqx.tree_at(
        lambda e: e.quadrature.nodes_nm, exposure, jnp.array([[550.0] * 3])
    )
    assert np.all(
        np.isnan(eqx.filter_jit(lambda e: photon_rates(scene, path, e))(outside).planet)
    )
