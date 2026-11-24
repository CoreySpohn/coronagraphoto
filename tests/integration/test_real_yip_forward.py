"""Full FITS scene/YIP observation integration using small synthetic inputs."""

import equinox as eqx
import jax
import numpy as np
import pytest
from conftest import exposure_for, make_path, write_scene, write_yip
from optixstuff import FramePlan
from optixstuff.spectral import Band, Quadrature
from skyscapes import from_exovista
from yippy import open_yip

from coronagraphoto import expected, observe


@pytest.mark.medium(reason="FITS scene and YIP loaders plus observation compilation")
def test_fits_scene_yip_observation(tmp_path):
    """Fails on incompatible scene, YIP, detector or observation adapters.

    Both loaders consume genuine FITS products. Deterministic rates must be
    positive for a planet inside the synthetic table's source nodes (0.35 AU is
    2.6 lambda/D, the nodes fill a square of half-width 3) and exactly zero
    for one beyond them (1 AU is 21), which the table does not model; draw
    equality is bitwise because identical keys call one sampler.
    """
    scene = from_exovista(
        write_scene(tmp_path / "scene.fits", a_au=0.35), load_disk=False
    )
    coro = open_yip(write_yip(tmp_path / "yip")).prepare()
    path = eqx.tree_at(lambda p: p.coronagraph, make_path(n=24, step=0.5), coro)
    q = Quadrature.build((Band(545.0, 555.0),), scene, path, order=1)
    plan = FramePlan(10.0, 1.0)
    exposure = exposure_for(scene, q, plan=plan)
    rates = expected(scene, path, exposure)
    assert float(rates.planet.sum()) > 0
    assert float(rates.local_zodi.sum()) > 0
    far = from_exovista(write_scene(tmp_path / "far.fits"), load_disk=False)
    beyond = expected(far, path, exposure)
    np.testing.assert_array_equal(beyond.planet, 0.0)
    assert float(beyond.local_zodi.sum()) > 0
    key = jax.random.PRNGKey(0)
    image = observe(key, scene, path, exposure)
    assert image.data.shape == (1, 24, 24)
    np.testing.assert_array_equal(image.data, path.detector.draw(key, rates, plan))


def scene_for(a_au):
    """One planet at ``a_au`` AU (0.05 arcsec per 0.5 AU at 10 pc)."""
    from conftest import make_system

    return make_system(separation_arcsec=a_au / 10)


def _yip_setup(tmp_path):
    """A synthetic YIP (source nodes to 3 lambda/D) on a detector inside its stamp."""
    coro = open_yip(write_yip(tmp_path / "yip")).prepare()
    path = eqx.tree_at(lambda p: p.coronagraph, make_path(n=24, step=0.5), coro)
    scene = scene_for(0.5)
    quadrature = Quadrature.build((Band(549.5, 550.5),), scene, path, order=1)
    return coro, path, exposure_for(scene, quadrature)


@pytest.mark.medium(reason="synthetic YIP load and two programs for 1 and 2 planets")
def test_planet_beyond_the_hull_leaves_the_inner_planet_untouched(tmp_path):
    """Fails if an unmodeled planet changes another planet's image.

    Seam anchor yippy to coronagraphoto. An inner planet at 2.6 lambda/D with a
    second planet at 3 times the node hull has the same image as the inner
    planet alone, to 1e-12 of its peak: the outer planet is flagged invalid
    for itself and contributes exact zero (before, a wrapped copy of its wing
    added 6e-9 or the whole image went NaN). Basis: floating-point, one
    summation in a different order.
    """
    import jax.numpy as jnp
    from conftest import make_system
    from skyscapes.physical_model import GridPhysicalModel
    from skyscapes.scene import Planet

    from coronagraphoto import photon_rates

    coro, path, exposure = _yip_setup(tmp_path)
    scale = float(coro.at_wavelength(550.0).source_grid.frame.lod_to_arcsec(550.0))
    hull = 4.25  # the farthest synthetic node, at (3, 3) lambda/D (a corner)
    inner, outer = 2.6 * scale, 3 * hull * scale

    def planets(separations):
        return make_system(
            separation_arcsec=jnp.array(separations),
            planet_groups=(
                Planet(
                    jnp.ones(len(separations)),
                    GridPhysicalModel(
                        jnp.array([400.0, 500.0, 600.0, 700.0]),
                        jnp.array([0.0, 90.0, 180.0]),
                        jnp.full((len(separations), 4, 3), 1e-10),
                    ),
                ),
            ),
        )

    one = np.asarray(photon_rates(planets([inner]), path, exposure).planet)
    two = np.asarray(photon_rates(planets([inner, outer]), path, exposure).planet)
    assert np.isfinite(two).all() and one.max() > 0
    np.testing.assert_allclose(two, one, rtol=0, atol=1e-12 * one.max())
    valid = coro.source_psfs(
        550.0, jnp.array([2.6, 3 * hull]), jnp.zeros(2), grid=None
    ).valid
    np.testing.assert_array_equal(valid, [True, False])


@pytest.mark.medium(reason="synthetic YIP load and reverse-mode programs")
def test_planet_position_gradient_is_exact_inside_and_zero_beyond_the_hull(tmp_path):
    """Fails if the planet image has a wrong or NaN gradient in its position.

    The summed planet image against the orbit's semi-major axis (the position
    at the epoch), checked with ``check_grads`` to order 1 in both modes at
    2.6 lambda/D (inside the nodes, off a node) and at 6.4 lambda/D (beyond
    the hull at 3), where the image is zero and the gradient is exactly
    zero. Fails on a square root of a zero node distance (NaN) or an
    unclamped read beyond the hull. The finite-difference probes of ``check_grads``
    are NumPy values, so each is converted to a JAX leaf before it enters the scene.
    """
    import jax
    import jax.numpy as jnp
    from jax.test_util import check_grads

    from coronagraphoto import photon_rates

    _, path, exposure = _yip_setup(tmp_path)

    def planet_sum(a_au):
        scene = eqx.tree_at(
            lambda s: s.dynamics.orbits.a_AU,
            scene_for(0.5),
            jnp.asarray(a_au)[None],
        )
        rates = photon_rates(scene, path, exposure)
        return rates.planet.sum() * 1e12

    with jax.enable_x64():
        import jax.numpy as jnp

        inside, beyond = jnp.asarray(0.5), jnp.asarray(1.2)
        assert float(planet_sum(inside)) > 0
        check_grads(
            planet_sum,
            (inside,),
            order=1,
            modes=("fwd", "rev"),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )
        assert float(planet_sum(beyond)) == 0.0
        gradient = jax.grad(planet_sum)(beyond)
        assert float(gradient) == 0.0
