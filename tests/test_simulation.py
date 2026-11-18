"""Photon reader radiometry and JAX contracts."""

import numpy as np
import pytest


def test_worked_example():
    """Fails if the reader omits a spectral factor; 1e-9 quadrature bound."""
    import coronagraphoto as cp

    assert hasattr(cp, "image_rates"), "the shared photon reader must be public"
    from conftest import worked_setup
    from hwoutils.grids import SkyOrientation
    from hwoutils.spectral import Band, Quadrature
    from numpy.polynomial.legendre import leggauss

    system, path = worked_setup()
    bands = (Band(500, 600),)
    rates = cp.image_rates(system, path, bands, time_d=0.0, pointing=SkyOrientation(0))
    quad = Quadrature.build(bands, None, 4)
    actual = float(rates.integrate(quad).planet.sum())
    electrons = float(path.detector.electrons(rates, quad).planet.sum())
    nodes, weights = leggauss(128)
    wave, weights = 550 + 50 * nodes, 50 * weights
    # Conventional 3631 Jy AB zero point and SI Planck constant.
    photons = 3631.0 * 10 ** (-0.4 * 5) * 1e-26 / (6.62607015e-34 * wave)
    contrast = 1e-10 * (1 - 0.004 * (wave - 550))
    tau = 0.30 + 0.0006 * (wave - 550)
    primitive = 40 * 0.40 * photons * contrast * tau
    expected = np.sum(weights * primitive)
    expected_e = np.sum(weights * primitive * 0.9 * (0.80 - 0.002 * (wave - 550)))
    # Gaussian truncation below 1e-15; dense cubic stellar table below 1e-11.
    np.testing.assert_allclose([actual, electrons], [expected, expected_e], rtol=1e-9)
    np.testing.assert_allclose(
        [expected, expected_e], [0.0477819, 0.0346767], atol=5e-8, rtol=0
    )
    print(
        f"worked photons={actual:.12g} electrons={electrons:.12g} "
        f"independent={expected:.12g},{expected_e:.12g}"
    )


@pytest.mark.parametrize("chromatic", [False, True])
def test_node_yield_once(chromatic):
    """Fails on double yield or band-mean QE; exact algebra at 32 float64 eps."""
    from conftest import worked_setup
    from hwoutils.grids import SkyOrientation
    from hwoutils.spectral import Band, Quadrature

    from coronagraphoto import image_electrons, image_rates

    system, path = worked_setup()
    bands = (Band(500, 600),)
    q = Quadrature.build(bands, None, 4)
    rates = image_rates(system, path, bands, time_d=0.0, pointing=SkyOrientation(0))
    if not chromatic:
        import equinox as eqx
        import jax.numpy as jnp

        path = eqx.tree_at(
            lambda p: p.detector._qe_curve,
            path,
            (jnp.array([500.0, 600.0]), jnp.array([0.8, 0.8])),
        )
    actual = image_electrons(
        system, path, bands, time_d=0.0, pointing=SkyOrientation(0)
    ).planet
    expected = np.zeros_like(actual)
    doubled = np.zeros_like(actual)
    for k, (wave, weight) in enumerate(
        zip(np.asarray(q.nodes_nm[0]), np.asarray(q.weights_nm[0]), strict=True)
    ):
        eta = 0.9 * (0.8 - (0.002 * (wave - 550) if chromatic else 0.0))
        expected += eta * weight * np.asarray(rates.planet[:, k])
        doubled += eta**2 * weight * np.asarray(rates.planet[:, k])
    np.testing.assert_allclose(
        actual, expected, rtol=32 * np.finfo(float).eps, atol=1e-18
    )
    # Mutation controls on the same observable: >1% and >0.1% respectively.
    assert abs(doubled.sum() / expected.sum() - 1) > 0.01
    band_mean = 0.9 * 0.8 * rates.integrate(q).planet.sum()
    if chromatic:
        assert abs(float(band_mean) / expected.sum() - 1) > 0.001
    else:
        np.testing.assert_allclose(
            band_mean, expected.sum(), rtol=32 * np.finfo(float).eps
        )


@pytest.mark.medium(reason="two bands, reverse-mode scan and JIT compilation")
def test_streaming_jit_and_gradient():
    """Fails on incorrect scan band indexing or broken contrast gradients.

    Scan/eager tolerance 64 eps; central difference h=1e-4 has a 1e-9
    relative roundoff allowance for the linear contrast response.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from conftest import worked_setup
    from hwoutils.grids import SkyOrientation
    from hwoutils.spectral import Band, Quadrature

    from coronagraphoto import image_electrons, image_rates

    system, path = worked_setup()
    bands = (Band(500, 540), Band(550, 600))
    pointing = SkyOrientation(17.0)
    options = dict(time_d=jnp.asarray(0.0), pointing=pointing)

    def render(s):
        return image_electrons(s, path, bands, **options)

    eager = render(system)
    full = path.detector.electrons(
        image_rates(system, path, bands, **options), Quadrature.build(bands, None, 4)
    )
    compiled = eqx.filter_jit(render)(system)
    for a, b, c in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(full),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=64 * np.finfo(float).eps, atol=1e-18)
        np.testing.assert_allclose(a, c, rtol=64 * np.finfo(float).eps, atol=1e-18)

    def total(scale):
        varied = eqx.tree_at(
            lambda s: s.planets[0].physical_model.contrast_grid,
            system,
            system.planets[0].physical_model.contrast_grid * scale,
        )
        return render(varied).planet.sum()

    h = 1e-4
    np.testing.assert_allclose(
        jax.grad(total)(1.0), (total(1 + h) - total(1 - h)) / (2 * h), rtol=1e-9
    )


def test_refused_planet_psf_only_invalidates_lit_source():
    """A refused PSF contributes zero for a dark planet and NaN for a lit one."""
    from dataclasses import replace

    import equinox as eqx
    import jax.numpy as jnp
    from conftest import GaussianCoronagraph, make_path, make_system
    from hwoutils.grids import SkyOrientation
    from hwoutils.spectral import Band
    from optixstuff import SourcePSFs

    from coronagraphoto import image_rates

    class RefusedSource(GaussianCoronagraph):
        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            return SourcePSFs(
                replace(psfs.image, data=jnp.full_like(psfs.image.data, jnp.nan)),
                jnp.zeros_like(psfs.valid),
            )

    path = make_path(n=8)
    path = eqx.tree_at(lambda p: p.coronagraph, path, RefusedSource(n=8))
    for contrast in (0.0, 1e-10):
        planet = image_rates(
            make_system(contrast=contrast),
            path,
            (Band(545.0, 555.0),),
            time_d=0.0,
            pointing=SkyOrientation(0),
            quadrature_order=1,
        ).planet
        if contrast == 0.0:
            np.testing.assert_array_equal(planet, 0.0)
        else:
            assert np.isnan(planet).all()
