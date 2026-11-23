"""Photon reader radiometry and JAX contracts."""

import numpy as np
import pytest


@pytest.mark.medium(reason="1600 declared stellar nodes in the worked spectrum")
@pytest.mark.fail_slow("90s")  # over 30 s on the hosted runner
def test_worked_example():
    """Fails if the reader omits a spectral factor; 1e-9 quadrature bound."""
    import coronagraphoto as cp

    assert hasattr(cp, "photon_rates"), "the shared photon reader must be public"
    from conftest import worked_setup
    from numpy.polynomial.legendre import leggauss
    from optixstuff import Exposure
    from optixstuff.spectral import Band, Quadrature

    system, path = worked_setup()
    bands = (Band(500, 600),)
    q = Quadrature.build(bands, system, path)
    scene, exposure = system, Exposure(time_d=0.0, roll_deg=0, quadrature=q)
    rates = cp.photon_rates(scene, path, exposure)
    actual = float(rates.integrate().planet.sum())
    electrons = float(path.detector.electrons(rates).planet.sum())
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
    from optixstuff import Exposure
    from optixstuff.spectral import Band, Quadrature

    import coronagraphoto as cp

    # The yield identity needs a color slope and an interior break.
    system, path = worked_setup(n_wavelengths=7)
    bands = (Band(500, 600),)
    q = Quadrature.build(bands, system, path)
    scene, exposure = system, Exposure(time_d=0.0, roll_deg=0, quadrature=q)
    rates = cp.photon_rates(scene, path, exposure)
    if not chromatic:
        import equinox as eqx
        import jax.numpy as jnp

        path = eqx.tree_at(
            lambda p: p.detector._qe_curve,
            path,
            (jnp.array([500.0, 600.0]), jnp.array([0.8, 0.8])),
        )
    actual = cp.expected(scene, path, exposure).planet
    expected = np.zeros_like(actual)
    doubled = np.zeros_like(actual)
    for k, (wave, weight) in enumerate(
        zip(
            np.asarray(q.nodes_nm[0]), np.asarray(q.weights("response")[0]), strict=True
        )
    ):
        eta = 0.9 * (0.8 - (0.002 * (wave - 550) if chromatic else 0.0))
        expected += eta * weight * np.asarray(rates.planet[:, k])
        doubled += eta**2 * weight * np.asarray(rates.planet[:, k])
    np.testing.assert_allclose(
        actual, expected, rtol=32 * np.finfo(float).eps, atol=1e-18
    )
    # Mutation controls on the same observable: >1% and >0.1% respectively.
    assert abs(doubled.sum() / expected.sum() - 1) > 0.01
    band_mean = 0.9 * 0.8 * rates.integrate().planet.sum()
    if chromatic:
        assert abs(float(band_mean) / expected.sum() - 1) > 0.001
    else:
        np.testing.assert_allclose(
            band_mean, expected.sum(), rtol=32 * np.finfo(float).eps
        )


@pytest.mark.medium(reason="two bands, reverse-mode scan and JIT compilation")
@pytest.mark.fail_slow("90s")  # over 30 s on the hosted runner
def test_streaming_jit_and_gradient():
    """Fails on incorrect scan band indexing or broken contrast gradients.

    Scan/eager tolerance 64 eps; central difference h=1e-4 has a 1e-9
    relative roundoff allowance for the linear contrast response.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from conftest import worked_setup
    from optixstuff import Exposure
    from optixstuff.spectral import Band, Quadrature

    import coronagraphoto as cp

    # Two bands with unequal node counts exercise zero-weight padding under JIT.
    system, path = worked_setup(n_wavelengths=16)
    bands = (Band(500, 540), Band(550, 600))
    q = Quadrature.build(bands, system, path)
    exposure = Exposure(time_d=jnp.asarray(0.0), roll_deg=17.0, quadrature=q)

    def render(s):
        return cp.expected(s, path, exposure)

    eager = render(system)
    full = path.detector.electrons(cp.photon_rates(system, path, exposure))
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
    """A refused PSF contributes zero, lit or dark, and never NaN.

    A source outside the source domain has a finite zero image and is flagged
    for that source alone.
    """
    from dataclasses import replace

    import equinox as eqx
    import jax.numpy as jnp
    from conftest import GaussianCoronagraph, make_path, make_system
    from optixstuff import Exposure, SourcePSFs
    from optixstuff.spectral import Quadrature

    import coronagraphoto as cp

    class RefusedSource(GaussianCoronagraph):
        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            return SourcePSFs(
                replace(psfs.image, data=jnp.zeros_like(psfs.image.data)),
                jnp.zeros_like(psfs.valid),
            )

    path = make_path(n=8)
    path = eqx.tree_at(lambda p: p.coronagraph, path, RefusedSource(n=8))
    for contrast in (0.0, 1e-10):
        planet = cp.photon_rates(
            make_system(contrast=contrast),
            path,
            Exposure(
                time_d=0.0, roll_deg=0, quadrature=Quadrature.monochromatic(550.0)
            ),
        ).planet
        np.testing.assert_array_equal(planet, 0.0)


def test_expected_integrates_declared_response_kink():
    """Fails if the image integrates across a declared triangular-filter kink.

    Flat local sky and constant yield isolate the spectral integral at the
    central pixel. A single unsplit midpoint would overestimate the area.
    """
    import equinox as eqx
    import jax.numpy as jnp
    from conftest import make_path, make_system, with_zodi
    from hwoutils.constants import arcsec2rad
    from optixstuff import Exposure, TransmissiveElement
    from optixstuff.spectral import Band, Quadrature, Transmission
    from skyscapes import PrecomputedZodi

    import coronagraphoto as cp

    band = Band(510.0, 590.0)
    response = Transmission([510.0, 533.0, 590.0], [0.0, 1.0, 0.0])
    system, path = make_system(), make_path(n=8)
    path = eqx.tree_at(
        lambda p: p.attenuating_elements, path, (TransmissiveElement(response),)
    )
    path = eqx.tree_at(
        lambda p: p.detector._qe_curve,
        path,
        (jnp.array([500.0, 600.0]), jnp.full(2, 0.8)),
    )
    sky = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    q = Quadrature.build((band,), system, path, sky, order=1)
    image = cp.expected(
        with_zodi(system, sky), path, Exposure(time_d=0.0, roll_deg=0, quadrature=q)
    )
    omega = (path.detector.pixel_scale_arcsec * arcsec2rad) ** 2
    expected = 40 * 0.3 * 1e12 * omega * 0.8 * 0.9 * band.width_nm / 2
    np.testing.assert_allclose(
        image.local_zodi[0, 4, 4], expected, rtol=32 * np.finfo(float).eps, atol=0
    )


def test_source_scales_are_linear_in_the_image_and_live_for_gradients():
    """Fails if a source's flux_scale misses its image or leaks into another.

    Basis: each rate is a product of float64 factors, so a scale multiplies
    its rates by exactly its factor to 1e-14 relative. The star scale reaches
    the leakage and the planet, the planet scale the planet, the disk scale the
    exozodi and the zodi scale the local zodi. The set is bilinear in star and
    planet, not jointly linear. The leaf is live: the gradient of the rate
    with respect to it is the rate.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from conftest import make_path, make_system, with_zodi
    from optixstuff.spectral import Band
    from skyscapes import AYOZodi
    from skyscapes.disk import AYOExozodi

    import coronagraphoto as cp

    scene = with_zodi(make_system(disk=AYOExozodi(level_zodis=3.0)), AYOZodi())
    path = make_path(n=8, cube=True)
    exposure = cp.build_exposure(scene, path, (Band(500.0, 600.0),), time_d=0.0)
    names = ("planet", "star_leakage", "local_zodi", "exozodi")
    where = dict(
        star=lambda s: s.star.flux_scale,
        planet=lambda s: s.planets[0].flux_scale,
        disk=lambda s: s.disk.flux_scale,
        zodi=lambda s: s.local_zodi.flux_scale,
    )

    def rates(**scales):
        varied = scene
        for source, value in scales.items():
            current = where[source](scene)
            varied = eqx.tree_at(where[source], varied, jnp.full_like(current, value))
        return cp.photon_rates(varied, path, exposure)

    base = rates()
    for scales, factors in (
        (dict(star=2.0), (2.0, 2.0, 1.0, 1.0)),
        (dict(planet=3.0), (3.0, 1.0, 1.0, 1.0)),
        (dict(disk=5.0), (1.0, 1.0, 1.0, 5.0)),
        (dict(zodi=7.0), (1.0, 1.0, 7.0, 1.0)),
        (dict(star=2.0, planet=3.0), (6.0, 2.0, 1.0, 1.0)),
    ):
        scaled = rates(**scales)
        for name, factor in zip(names, factors, strict=True):
            np.testing.assert_allclose(
                getattr(scaled, name),
                factor * getattr(base, name),
                rtol=1e-14,
                atol=0.0,
                err_msg=f"{scales} on {name}",
            )
    for name in names:
        assert float(jnp.max(jnp.abs(getattr(base, name)))) > 0.0

    for source, name in (
        ("star", "star_leakage"),
        ("planet", "planet"),
        ("disk", "exozodi"),
        ("zodi", "local_zodi"),
    ):

        def total(scale, source=source, name=name):
            varied = eqx.tree_at(
                where[source], scene, jnp.full_like(where[source](scene), scale)
            )
            return getattr(cp.photon_rates(varied, path, exposure), name).sum()

        np.testing.assert_allclose(
            jax.grad(total)(1.0), float(getattr(base, name).sum()), rtol=1e-12
        )


def test_disk_render_does_not_depend_on_detector_extent():
    """Fails if disk light off the detector is dropped; 1e-10 relative to peak.

    A small detector's exozodi equals the crop of a detector that spans the
    whole coronagraph grid, for the same scene and a disk wider than the small
    detector. Sky just off the small detector scatters onto it, so the
    radiance must be evaluated on the coronagraph source grid, not on the
    detector. Evaluating it on the detector grid gave a crop sum of 0.9919 of
    the full render's here, worst pixel error 1.6e-3 of peak (0.8905 and 68 %
    for an 8 lambda/D detector under a 20 lambda/D disk). Basis: both paths
    sample the same float64 radiance on matching grids, so they agree to
    rounding well inside 1e-10.
    """
    from conftest import make_path, make_system
    from optixstuff.spectral import Band
    from skyscapes.disk import AYOExozodi

    import coronagraphoto as cp

    scene = make_system(disk=AYOExozodi(level_zodis=3.0))
    bands = (Band(500.0, 600.0),)

    def exozodi(detector_n):
        path = make_path(n=64, cube=True, detector_n=detector_n)
        exposure = cp.build_exposure(scene, path, bands, time_d=0.0)
        return np.asarray(cp.photon_rates(scene, path, exposure).exozodi)[0]

    small, full = exozodi(32), exozodi(64)
    crop = full[:, 16:48, 16:48]
    assert np.isfinite(crop).all() and crop.max() > 0
    print(f"sum ratio {small.sum() / crop.sum():.4f}")
    print(f"worst/peak {np.abs(small - crop).max() / crop.max():.3e}")
    np.testing.assert_allclose(small, crop, rtol=0, atol=1e-10 * crop.max())


def test_pixels_beyond_the_tables_are_zero_and_unsupported():
    """Fails if light is invented beyond the tables or support ignores a node.

    A detector twice the table: 48 pixels of 0.25 lambda/D against a 24 pixel
    table (half-width 3 lambda/D, linear products reach in by half a table
    pixel). The table is fixed in lambda/D, so its angular footprint shrinks
    at the blue node. Expected support is computed from primitives: a pixel
    is supported when its far edge lies inside 2.875 lambda/D at the node's
    own scale, at every node. Star leakage and local zodi are exactly zero
    outside, including the pixels the red node alone reaches; inside, the
    uniform sky and star are untouched. Basis: exact.
    """
    import jax.numpy as jnp
    from conftest import make_path, make_system, with_zodi
    from optixstuff.spectral import Band
    from skyscapes import PrecomputedZodi

    import coronagraphoto as cp

    n, step = 24, 0.25
    path = make_path(n=n, step=step, detector_n=2 * n)
    zodi = PrecomputedZodi(jnp.array([400.0, 700.0]), jnp.full(2, 1e12))
    scene = with_zodi(make_system(), zodi)
    bands = (Band(500.0, 600.0),)
    exposure = cp.build_exposure(scene, path, bands, time_d=0.0, order=2)
    nodes = np.asarray(exposure.quadrature.nodes_nm)[0]
    assert nodes.size >= 2
    rates = cp.expected(scene, path, exposure)
    photons = cp.photon_rates(scene, path, exposure)

    pixel_arcsec = float(path.detector.pixel_scale_arcsec)
    lod = np.asarray(
        [float(path.coronagraph.grid.frame.lod_to_arcsec(w)) for w in nodes]
    )
    centre = (2 * n - 1) / 2
    far_edge = (np.abs(np.arange(2 * n) - centre) + 0.5) * pixel_arcsec
    # Table half-width 3 lambda/D shrunk by the linear reach of half a table pixel.
    reach = (n * step / 2 - 0.5 * step) * lod[:, None]
    per_axis = far_edge[None, :] <= reach
    support = per_axis[:, :, None] & per_axis[:, None, :]
    expected_support = support.all(axis=0)
    blue_only_misses = support[-1] & ~support[0]
    assert blue_only_misses.any(), (
        "the band must have a pixel only the red node reaches"
    )

    for rec in (rates, photons):
        assert rec.supported.shape == (1, 2 * n, 2 * n)
        np.testing.assert_array_equal(rec.supported[0], expected_support)
        for name in ("star_leakage", "local_zodi", "planet", "exozodi"):
            outside = np.asarray(getattr(rec, name))[..., ~expected_support]
            np.testing.assert_array_equal(outside, 0.0)
    inside = np.asarray(rates.star_leakage)[0][expected_support]
    assert np.all(inside > 0)
    assert np.all(np.asarray(rates.local_zodi)[0][expected_support] > 0)
    assert np.isfinite(np.asarray(rates.star_leakage)).all()
    integrated = photons.integrate()
    np.testing.assert_array_equal(integrated.supported, photons.supported)
