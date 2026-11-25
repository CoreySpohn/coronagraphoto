"""Photon reader radiometry and JAX contracts."""

import numpy as np
import pytest
from conftest import GaussianCoronagraph, IntegratedGaussianCoronagraph


@pytest.mark.medium(reason="1600 declared stellar nodes in the worked spectrum")
@pytest.mark.fail_slow("90s")  # over 30 s on the hosted runner
def test_worked_example():
    """Fails if the reader omits a spectral factor; 1e-9 quadrature bound."""
    import coronagraphoto as cp

    assert hasattr(cp, "photon_rates"), "the shared photon reader must be public"
    from conftest import exposure_for, worked_setup
    from numpy.polynomial.legendre import leggauss
    from optixstuff.spectral import Band, Quadrature

    system, path = worked_setup()
    bands = (Band(500, 600),)
    q = Quadrature.build(bands, system, path)
    scene, exposure = system, exposure_for(system, q)
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
    from conftest import exposure_for, worked_setup
    from optixstuff.spectral import Band, Quadrature

    import coronagraphoto as cp

    # The yield identity needs a color slope and an interior break.
    system, path = worked_setup(n_wavelengths=7)
    bands = (Band(500, 600),)
    q = Quadrature.build(bands, system, path)
    scene, exposure = system, exposure_for(system, q)
    rates = cp.photon_rates(scene, path, exposure)
    if not chromatic:
        import equinox as eqx
        import jax.numpy as jnp

        path = eqx.tree_at(
            lambda p: p.detector._qe_values,
            path,
            jnp.array([0.8, 0.8]),
        )
    actual = cp.expected(scene, path, exposure).planet
    expected = np.zeros_like(actual)
    doubled = np.zeros_like(actual)
    for k, (wave, weight) in enumerate(
        zip(np.asarray(q.nodes_nm[0]), np.asarray(q.weights()[0]), strict=True)
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
def test_streaming_jit_and_gradient(precision):
    """Fails on incorrect scan band indexing or broken contrast gradients.

    Scan/eager tolerance 64 eps; central difference h=1e-4 has a 1e-9
    relative roundoff allowance for the linear contrast response.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from conftest import exposure_for, worked_setup
    from optixstuff.spectral import Band, Quadrature

    import coronagraphoto as cp

    # Two bands with unequal node counts exercise zero-weight padding under JIT.
    system, path = worked_setup(n_wavelengths=16)
    bands = (Band(500, 540), Band(550, 600))
    q = Quadrature.build(bands, system, path)
    exposure = exposure_for(system, q, time_d=jnp.asarray(0.0), roll_deg=17.0)

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

    if precision == np.float64:
        h = 1e-4
        np.testing.assert_allclose(
            jax.grad(total)(1.0), (total(1 + h) - total(1 - h)) / (2 * h), rtol=1e-9
        )


def _position_grad_setup(knots):
    """Two planets whose sky offsets follow ``a_AU``, on a ``knots``-knot table.

    Returns ``(scene_of, path, exposure)``: ``scene_of(a)`` is the scene with
    the planets' semi-major axes ``a`` in AU (their sky positions scale with
    it), real constants, in the Gaussian coronagraph's field.
    """
    import equinox as eqx
    import jax.numpy as jnp
    from conftest import Planet, exposure_for, make_path, make_system
    from optixstuff.spectral import Band, Quadrature
    from skyscapes.physical_model import GridPhysicalModel

    wave = jnp.linspace(400.0, 700.0, knots)
    base = make_system(
        wavelengths_nm=wave,
        separation_arcsec=jnp.array([0.02, 0.04]),
        sky_pa_deg=30.0,
        planet_groups=(
            Planet(
                jnp.ones(2),
                GridPhysicalModel(
                    wave, jnp.array([0.0, 90.0, 180.0]), jnp.full((2, knots, 3), 1e-10)
                ),
            ),
        ),
    )
    path = make_path(n=40)
    exposure = exposure_for(base, Quadrature.build((Band(500.0, 600.0),), base, path))

    def scene_of(a):
        return eqx.tree_at(lambda s: s.dynamics.orbits.a_AU, base, a)

    return scene_of, path, exposure


def _position_grads(knots):
    """Reverse gradients of the planet image sum with respect to ``a_AU``.

    Returns the photon-reader gradient (a ``lax.map`` over nodes), the
    streamed-electron gradient (a ``scan``) and the compiled temp bytes of each.
    """
    import jax
    import jax.numpy as jnp

    import coronagraphoto as cp

    scene_of, path, exposure = _position_grad_setup(knots)
    a = jnp.array([0.2, 0.4])
    out = []
    for read in (cp.photon_rates, cp.expected):

        def total(x, read=read):
            return read(scene_of(x), path, exposure).planet.sum()

        grad = jax.jit(jax.grad(total))
        temp = grad.lower(a).compile().memory_analysis().temp_size_in_bytes
        out.append((grad(a), temp))
    return exposure.quadrature.nodes_nm.shape[1], out


@pytest.mark.medium(reason="reverse-mode compiles at two node counts, with and without")
def test_node_checkpoint_keeps_reverse_gradients_and_flat_memory(monkeypatch):
    """Fails if the node render is unchecked or the checkpoint changes a gradient.

    Truth: the unwrapped node function (``jax.checkpoint`` replaced by the
    identity) is the same expression, so the reverse gradients with respect to
    the planets' sky positions agree to floating point, rtol 1e-12 (measured:
    bitwise equal). Memory: the compiled temp of each reverse gradient at 12
    nodes is under 1.3 times that at 4 (mutation basis: unwrapped it grows by
    2.43 times for the photon reader and 2.38 for the streamed electrons;
    wrapped 1.15 and 1.02, the photon reader also returning every node's
    image).
    """
    import jax

    small_nodes, small = _position_grads(4)
    large_nodes, large = _position_grads(8)
    assert large_nodes == 3 * small_nodes
    for (_, t_small), (g_large, t_large) in zip(small, large, strict=True):
        assert t_large < 1.3 * t_small, t_large / t_small
        assert float(abs(g_large).max()) > 0.0

    monkeypatch.setattr(jax, "checkpoint", lambda f, **kwargs: f)
    jax.clear_caches()
    _, plain = _position_grads(4)
    plain_ratio = [
        p_large[1] / p_small[1]
        for p_small, p_large in zip(plain, _position_grads(8)[1], strict=True)
    ]
    assert min(plain_ratio) > 2.0
    for (wrapped, _), (unwrapped, _) in zip(small, plain, strict=True):
        np.testing.assert_allclose(
            wrapped,
            unwrapped,
            rtol=1e-12,
            err_msg=f"gradients differ by {abs(wrapped - unwrapped).max()}",
        )
    jax.clear_caches()


class BoundedGaussianCoronagraph(GaussianCoronagraph):
    """Gaussian PSFs with source support inside the native square grid."""

    def _native_source_psfs(self, x_lod, y_lod):
        from dataclasses import replace

        import jax.numpy as jnp
        from optixstuff import SourcePSFs

        psfs = super()._native_source_psfs(x_lod, y_lod)
        half_extent = self.grid.shape[0] * self.grid.step / 2
        valid = (jnp.abs(x_lod) <= half_extent) & (jnp.abs(y_lod) <= half_extent)
        return SourcePSFs(
            replace(
                psfs.image, data=jnp.where(valid[:, None, None], psfs.image.data, 0)
            ),
            valid,
        )


def test_refused_planet_psf_only_invalidates_lit_source():
    """Outside the square source domain, the support flag and image are zero.

    The manufactured grid reaches only 1 lambda/D along each axis, less than
    the planet's north offset. Admission and discarded flux are exact zeros.
    """
    from dataclasses import replace

    from conftest import exposure_for, make_path, make_system
    from optixstuff.spectral import Quadrature

    import coronagraphoto as cp

    path = replace(make_path(n=8), coronagraph=BoundedGaussianCoronagraph(n=8))
    for contrast in (0.0, 1e-10):
        scene = make_system(contrast=contrast, smooth_planet=True)
        exposure = exposure_for(scene, Quadrature.monochromatic(550.0))
        np.testing.assert_array_equal(
            cp.photon_rates(scene, path, exposure).planet, 0.0
        )
        np.testing.assert_array_equal(
            cp.planet_source_support(scene, path, exposure), False
        )


def test_planet_source_support_uses_exposure_roll_and_times():
    """Match direct source PSF flags at rolled scene positions, bit for bit.

    Boolean admission needs no tolerance. The middle planet crosses the square
    boundary between epochs and also changes admission under a zero roll.
    """
    from dataclasses import replace

    import jax.numpy as jnp
    from conftest import FlatContrast, exposure_for, make_path, make_system
    from optixstuff.grids import SkyOrientation
    from optixstuff.spectral import Band, Quadrature
    from skyscapes.scene import Planet

    import coronagraphoto as cp

    coro = BoundedGaussianCoronagraph(n=16)
    path = replace(make_path(n=16), coronagraph=coro)
    separations = coro.frame.lod_to_arcsec(550.0) * jnp.array([1.0, 2.4, 4.0])
    scene = make_system(
        separation_arcsec=separations,
        planet_groups=(Planet(jnp.ones(3), FlatContrast(jnp.asarray(1e-10))),),
    )
    times, roll = jnp.array([0.0, 10.0]), 45.0
    bands = (Band(520.0, 540.0), Band(560.0, 580.0))
    rule = Quadrature.build(bands, order=1)
    exposure = exposure_for(scene, rule, time_d=times, roll_deg=roll)
    east, north = scene.positions(times)

    def direct_flags(angle):
        x, y = SkyOrientation(angle).sky_to_detector(east, north)
        masks = []
        for band in bands:
            center = band.center_nm
            scale = coro.at_wavelength(center).source_grid.frame.lod_to_arcsec(center)
            flags = coro.source_psfs(center, x.ravel() / scale, y.ravel() / scale).valid
            masks.append(flags.reshape(x.shape))
        return np.stack(masks)

    truth = direct_flags(roll)
    assert truth[:, 1, 0].all() and not truth[:, 1, 1].any()
    assert not direct_flags(0.0)[:, 1, 0].any()
    support = cp.planet_source_support(scene, path, exposure)
    assert isinstance(support, np.ndarray) and support.dtype == np.bool_
    assert support.shape == (2, 3, 2)
    np.testing.assert_array_equal(support, truth)

    empty = make_system(planet_groups=(), separation_arcsec=jnp.empty(0))
    support = cp.planet_source_support(empty, path, exposure)
    assert support.shape == (2, 0, 2) and support.dtype == np.bool_


@pytest.fixture
def multiband_source_domain():
    """A wide red component first in dispatch order and a narrow blue one."""
    from dataclasses import replace

    from conftest import make_path
    from optixstuff import MultiBandCoronagraph
    from optixstuff.spectral import Band

    red = BoundedGaussianCoronagraph(n=40, bandpass=Band(550.0, 600.0))
    blue = BoundedGaussianCoronagraph(n=16, bandpass=Band(500.0, 550.0))
    path = make_path(n=16, detector_n=8)
    return replace(path, coronagraph=MultiBandCoronagraph((red, blue)))


def test_planet_source_support_selects_each_component(
    multiband_source_domain, precision
):
    """Compare admission against the two square half-extents, exactly.

    At roll zero the planet is on the north axis. Its lambda/D separation at
    either band center is between the two grids' half-widths, so the flags
    must differ with no tolerance on the Boolean result.
    """
    from conftest import exposure_for, make_system
    from optixstuff.spectral import Band, Quadrature

    import coronagraphoto as cp

    path = multiband_source_domain
    bands = (Band(570.0, 580.0), Band(520.0, 530.0))
    separation = 0.06
    scene = make_system(separation_arcsec=separation, smooth_planet=True)
    exposure = exposure_for(scene, Quadrature.build(bands, path, order=1))
    half_extents = np.array([40, 16]) * 0.25 / 2
    wavelengths = np.array([band.center_nm for band in bands])
    offsets = separation / (np.rad2deg(wavelengths * 1e-9 / 6.0) * 3600)
    assert np.all((half_extents[1] < offsets) & (offsets < half_extents[0]))
    support = cp.planet_source_support(scene, path, exposure)
    np.testing.assert_array_equal(support[:, 0, 0], offsets <= half_extents)

    outside = exposure_for(scene, Quadrature.monochromatic(610.0))
    with pytest.raises(ValueError, match=r"610.*nm"):
        cp.planet_source_support(scene, path, outside)
    # A planet-free scene has nothing to select a component for.
    empty = make_system(planet_groups=(), separation_arcsec=np.empty(0))
    assert cp.planet_source_support(empty, path, outside).shape == (1, 0, 1)

    # In float32 a red edge 1e-5 nm above 550 nm rounds onto 550 nm, so the
    # renderer's node comparison admits red at a 550 nm node when its other
    # node holds red as well; the query compares in the node dtype among the
    # same components, not in the host's float64 over every component.
    from dataclasses import replace

    import jax.numpy as jnp
    from conftest import make_path
    from optixstuff import MultiBandCoronagraph

    if precision == np.float32:
        red = BoundedGaussianCoronagraph(n=40, bandpass=Band(550.00001, 600.0))
        blue = BoundedGaussianCoronagraph(n=16, bandpass=Band(500.0, 550.0))
        shifted = replace(
            make_path(n=16, detector_n=8),
            coronagraph=MultiBandCoronagraph((red, blue)),
        )
        scene = make_system(separation_arcsec=separation, smooth_planet=True)
        rule = Quadrature(
            (Band(540.0, 560.0),), jnp.array([[550.0, 555.0]]), jnp.ones((1, 2))
        )
        edge = exposure_for(scene, rule)
        planet = cp.photon_rates(scene, shifted, edge).planet
        support = cp.planet_source_support(scene, shifted, edge)
        assert float(planet[0, 0].sum()) > 0
        np.testing.assert_array_equal(support[:, 0, 0], [True])

    # A rule whose only node is red holds red alone, and the renderer reads a
    # sole held component at every node, so the query reads red at a center
    # (545 nm) that red's bandpass does not cover.
    scene = make_system(separation_arcsec=separation, smooth_planet=True)
    sole = exposure_for(
        scene, Quadrature((Band(530.0, 560.0),), np.array([[558.0]]), np.ones((1, 1)))
    )
    assert float(cp.photon_rates(scene, path, sole).planet.sum()) > 0
    np.testing.assert_array_equal(
        cp.planet_source_support(scene, path, sole)[:, 0, 0], [True]
    )


def test_planet_source_support_uses_band_center_for_crossing_band(
    multiband_source_domain,
):
    """Compare flags with the rendered lit planet for three manufactured bands.

    The blue component rejects the planet, so its flux is exactly zero. The
    red component admits it and produces positive flux. A crossing band with
    its center in red has red flux even though its first node is blue. The
    resulting Boolean comparison is exact.
    """
    from conftest import make_system
    from optixstuff.spectral import Band

    import coronagraphoto as cp

    path = multiband_source_domain
    scene = make_system(separation_arcsec=0.06, smooth_planet=True)
    bands = (Band(520.0, 530.0), Band(540.0, 580.0), Band(570.0, 580.0))
    exposure = cp.build_exposure(scene, path, bands, time_d=0.0, order=1, field_lod=0.0)
    assert exposure.quadrature.nodes_nm[1, 0] < 550.0 < bands[1].center_nm
    rates = cp.photon_rates(scene, path, exposure)
    planet = np.asarray(rates.integrate().planet)
    np.testing.assert_array_equal(planet[0], 0.0)
    totals = planet.sum(axis=(1, 2))
    assert np.all(totals[1:] > 0)
    support = cp.planet_source_support(scene, path, exposure)
    np.testing.assert_array_equal(support[:, 0, 0], totals > 0, err_msg=f"{totals=}")


def test_expected_integrates_declared_response_kink():
    """Fails if the image integrates across a declared triangular-filter kink.

    Flat local sky and constant yield isolate the spectral integral at the
    central pixel. A single unsplit midpoint would overestimate the area.
    """
    import equinox as eqx
    import jax.numpy as jnp
    from conftest import exposure_for, make_path, make_system, with_zodi
    from hwoutils.constants import arcsec2rad
    from optixstuff import TransmissiveElement
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
        lambda p: p.detector._qe_values,
        path,
        jnp.full(2, 0.8),
    )
    sky = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    q = Quadrature.build((band,), system, path, sky, order=1)
    scene = with_zodi(system, sky)
    image = cp.expected(scene, path, exposure_for(scene, q))
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
        # The two detectors declare different fields, hence different nodes; the
        # crop compares one layout, so the field is zero.
        exposure = cp.build_exposure(scene, path, bands, time_d=0.0, field_lod=0.0)
        return np.asarray(cp.photon_rates(scene, path, exposure).exozodi)[0]

    small, full = exozodi(32), exozodi(64)
    crop = full[:, 16:48, 16:48]
    assert np.isfinite(crop).all() and crop.max() > 0
    print(f"sum ratio {small.sum() / crop.sum():.4f}")
    print(f"worst/peak {np.abs(small - crop).max() / crop.max():.3e}")
    np.testing.assert_allclose(small, crop, rtol=0, atol=1e-10 * crop.max())


def test_disk_reaches_the_operator_on_its_scene_grid():
    """Fails if disk cells beyond the stored PSF grid never reach the operator.

    A coronagraph whose ``scene_grid`` pads its 24-pixel PSF grid by 8 cells
    a side (its source domain reaches past the stamp). The disk handed to
    ``render_scene`` must be on that grid, carry light in the pad, and equal
    the stamp-grid disk cell for cell inside the stamp. Before the disk was
    built on the stored PSF grid, so the pad was structurally empty. Basis:
    the interior cells integrate the same float64 radiance samples, so they
    agree to rounding (1e-12 relative); the pad holds 9.1 % of this disk's
    light (spike), bounded below at 5 %.
    """
    from dataclasses import replace

    import equinox as eqx
    import jax
    from conftest import GaussianSceneCoronagraph, make_path, make_system
    from optixstuff.spectral import Band
    from skyscapes.disk import AYOExozodi

    import coronagraphoto as cp

    n, pad = 24, 8
    seen = {}

    class Wide(GaussianSceneCoronagraph):
        @property
        def scene_grid(self):
            g = self.grid
            return replace(
                g,
                shape=(n + 2 * pad,) * 2,
                center_x=g.center_x + pad,
                center_y=g.center_y + pad,
            )

        def render_scene(self, scene, wavelength_nm, *, grid):
            seen.setdefault(type(self).__name__, scene)
            return super().render_scene(scene, wavelength_nm, grid=grid)

    class Stamp(GaussianSceneCoronagraph):
        def render_scene(self, scene, wavelength_nm, *, grid):
            seen.setdefault(type(self).__name__, scene)
            return super().render_scene(scene, wavelength_nm, grid=grid)

    # The operators are read as concrete values from outside the trace, which
    # holds under disable_jit because the node is not checkpointed there.
    scene = make_system(disk=AYOExozodi(level_zodis=3.0))
    bands = (Band(500.0, 600.0),)
    base = make_path(n=n, cube=True)
    for model in (Wide, Stamp):
        path = eqx.tree_at(lambda p: p.coronagraph, base, model(n, 0.25))
        with jax.disable_jit():
            exposure = cp.build_exposure(scene, path, bands, time_d=0.0, order=1)
            cp.photon_rates(scene, path, exposure)

    wide, stamp = seen["Wide"], seen["Stamp"]
    assert wide.grid.shape == (n + 2 * pad,) * 2 and stamp.grid.shape == (n, n)
    data = np.asarray(wide.data)
    interior = data[pad:-pad, pad:-pad]
    outside = data.sum() - interior.sum()
    print(f"pad share of the disk {outside / data.sum():.3f}")
    assert outside > 0.05 * data.sum()
    np.testing.assert_allclose(interior, np.asarray(stamp.data), rtol=1e-12, atol=0)


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


@pytest.mark.parametrize(("cells", "detector_n"), [(3.0, 15), (3.0, 25), (2.4, 15)])
def test_support_follows_the_deposit_read_on_a_coarse_detector(cells, detector_n):
    """Fails if support uses the sampling reach where a pixel exceeds the table cell.

    A 24 cell table and a 48 cell table of the same smooth products, on one
    detector whose pixel is 3 or 2.4 table cells at 550 nm (more than 2 cells
    at the bluer nodes of the band, so they deposit). Truth is the render from
    the wide table, which reads no edge: wherever the narrow render is
    supported the two agree to 1e-12 relative (identical samples through the
    same linear read and band integral; the tolerance covers summation order).
    With the sampling margin the edge ring is called supported and differs
    there. Odd detectors put a pixel centre on the optical axis; at these
    sizes they are the geometries where the two margins call different pixels
    supported (a 2.5 cell pixel on an even detector happens to agree).
    """
    import equinox as eqx
    import jax.numpy as jnp
    from conftest import make_path, make_system, with_zodi
    from optixstuff.spectral import Band
    from skyscapes import PrecomputedZodi

    import coronagraphoto as cp

    step = 0.25
    zodi = PrecomputedZodi(jnp.array([400.0, 700.0]), jnp.full(2, 1e12))
    scene = with_zodi(make_system(), zodi)
    renders = {}
    for n in (24, 48):
        path = make_path(
            n=n, step=step, detector_n=detector_n, detector_step=cells * step
        )
        exposure = cp.build_exposure(
            scene, path, (Band(500.0, 600.0),), time_d=0.0, order=2
        )
        renders[n] = (exposure, eqx.filter_jit(cp.expected)(scene, path, exposure))
    nodes = np.asarray(renders[24][0].quadrature.nodes_nm)
    np.testing.assert_array_equal(nodes, np.asarray(renders[48][0].quadrature.nodes_nm))
    assert nodes.size >= 2
    narrow, wide = renders[24][1], renders[48][1]
    supported = np.asarray(narrow.supported)[0]
    assert supported.any() and not supported.all()
    assert np.asarray(wide.supported)[0][supported].all()
    for name in ("star_leakage", "local_zodi"):
        actual = np.asarray(getattr(narrow, name))[0][supported]
        truth = np.asarray(getattr(wide, name))[0][supported]
        assert np.all(truth > 0)
        np.testing.assert_allclose(actual, truth, rtol=1e-12, atol=0)


def test_returned_products_carry_the_measure_of_the_read():
    """Fails if a coronagraph returns an image whose measure differs from its read.

    Support takes the measure from the image the coronagraph returns on the
    detector grid, so a table coronagraph must return the measure it resampled
    with: cell integrals for the stellar map, intensive for the sky map.
    """
    from conftest import make_path
    from optixstuff.grids import Measure

    path = make_path(n=24, step=0.25, detector_n=15, detector_step=0.75)
    coro = path.coronagraph
    stellar = coro.stellar_map(550.0, 0.0, grid=path.detector.grid)
    sky = coro.uniform_sky_map(550.0, grid=path.detector.grid)
    assert stellar.measure == Measure.integral()
    assert sky.measure == Measure.intensive()


_TABLE_N = 32
_TABLE_STEP = 0.25
_SAMPLING_MARGIN = {"linear": 0.5, "cubic": 2.0, "cubic_spline": 2.0}
_DEPOSIT_HALF_SUPPORT = {"linear": 1.0, "cubic": 2.0}
# Pixels of the footprint read's set that the deposit rule leaves out, by
# geometry on the detectors below: one ring at 3 cells, none at 1 and 2.
_LOST_TO_DEPOSIT = {("linear", 0.75): 36, ("cubic", 0.75): 28}


def _declared_star(reconstruction, pad=0):
    """A table coronagraph whose star is a point-sampled density.

    The stellar map is a smooth positive field sampled at the cell centers of a
    ``_TABLE_N + 2 pad`` cell table and declared ``density("grid")`` with
    ``reconstruction``. The inner ``_TABLE_N`` cells hold the same samples for
    every ``pad``; the extra ring holds a different, much brighter field.
    """
    import equinox as eqx
    import jax.numpy as jnp
    from conftest import GaussianCoronagraph
    from optixstuff.grids import Image, Measure

    class DeclaredStar(GaussianCoronagraph):
        stellar_reconstruction: str = eqx.field(static=True)
        half_width: float = eqx.field(static=True)

        def __init__(self):
            super().__init__(_TABLE_N + 2 * pad, _TABLE_STEP)
            self.stellar_reconstruction = reconstruction
            self.half_width = _TABLE_N * _TABLE_STEP / 2

        def _native_stellar_map(self, stellar_diameter_lod):
            x, y = self.grid.coordinates()
            field = 1e-9 * (1.5 + jnp.cos(0.9 * x + 0.3) * jnp.cos(0.7 * y - 0.2))
            inside = (jnp.abs(x) < self.half_width) & (jnp.abs(y) < self.half_width)
            data = jnp.where(inside, field, 40.0 * field + 3e-8)
            return Image(
                data,
                self.grid,
                (),
                Measure.density("grid"),
                "stellar_flux_fraction",
                self.stellar_reconstruction,
            )

    return DeclaredStar()


def _supported_from_corners(detector_n, pixel_cells, margin_cells):
    """Detector pixels whose four corners lie ``margin_cells`` inside the table.

    Pure geometry on a detector and a table that share their center and axes:
    a pixel's far edge, in table cells, against the table half-width shrunk by
    the margin. The 1e-9 cell slack is the one ``coverage`` allows.
    """
    centre = (detector_n - 1) / 2
    far_edge = (np.abs(np.arange(detector_n) - centre) + 0.5) * pixel_cells
    per_axis = far_edge <= _TABLE_N / 2 - margin_cells + 1e-9
    return per_axis[:, None] & per_axis[None, :]


@pytest.mark.parametrize("reconstruction", ["linear", "cubic", "cubic_spline"])
@pytest.mark.parametrize(
    ("pitch_lod", "detector_n"), [(0.25, 36), (0.5, 18), (0.75, 10)]
)
def test_star_support_matches_the_footprint_read(reconstruction, pitch_lod, detector_n):
    """Fails if the star's support is not the reach of the read on_grid performs.

    A density star with a declared reconstruction on a 32 cell table at 0.25
    lambda/D, read on detectors of 0.25, 0.5 and 0.75 lambda/D (1, 2 and 3
    table cells per pixel) that reach past the table, at one node at 550 nm.

    1. ``cubic_spline``: the supported set equals the pixels whose corners are
       2 cells inside the table. Truth: geometry, exact.
    2. ``linear`` and ``cubic``: the star term at every supported pixel is
       unchanged when the table is embedded in a wider one whose extra ring
       holds a field 40 times brighter (a finite-reach read does not see it).
       Truth: finite kernel reach. Tolerance: 64 eps relative, rounding only
       (the wide read forms its kernel positions from a shifted source center
       and sums over more columns, measured at most 3.8e-15, at 0.25 lambda/D;
       a pixel that reads the ring
       moves by order one). The supported set
       is the one the deposit rule gives, which is contained in the footprint
       read's own set (corners ``m`` cells inside: 0.5 linear, 2 cubic). On
       these detectors it is a ring narrower at 3 cells (36 pixels linear, 28
       cubic) and equal at 1 and 2 cells, where the two nodes straddle 2 cells
       and the bluer node sets both sets.
    3. Every supported pixel of the star term is positive: with no speckle
       excess the zero clip of the leakage does not bind.

    Fails for a ``coverage`` call that drops the reconstruction (margin 0:
    case 1 and the invariance of case 2), or a read whose reach exceeds the
    margin it reports.
    """
    import equinox as eqx
    from conftest import make_path, make_system
    from optixstuff.spectral import Band

    import coronagraphoto as cp

    scene = make_system()
    bands = (Band(549.5, 550.5),)

    def render(pad):
        path = make_path(
            n=_TABLE_N, step=_TABLE_STEP, detector_n=detector_n, detector_step=pitch_lod
        )
        path = eqx.tree_at(
            lambda p: p.coronagraph, path, _declared_star(reconstruction, pad)
        )
        exposure = cp.build_exposure(scene, path, bands, time_d=0.0, order=1)
        return exposure, cp.expected(scene, path, exposure)

    exposure, narrow = render(0)
    nodes = np.asarray(exposure.quadrature.nodes_nm).ravel()
    # The table is fixed in lambda/D and the detector in arcsec, so a pixel
    # spans 550 / node of its 550 nm width; support is the AND over nodes.
    cells = [pitch_lod * 550.0 / float(w) / _TABLE_STEP for w in nodes]
    supported = np.asarray(narrow.supported)[0]
    leakage = np.asarray(narrow.star_leakage)[0]

    def expected(rule):
        sets = [_supported_from_corners(detector_n, g, rule(g)) for g in cells]
        return np.logical_and.reduce(sets)

    sampling = _SAMPLING_MARGIN[reconstruction]
    footprint_set = expected(lambda g: sampling)
    if reconstruction == "cubic_spline":
        np.testing.assert_array_equal(supported, footprint_set)
    else:
        k = _DEPOSIT_HALF_SUPPORT[reconstruction]

        def deposit(g):
            # The integral measure of the on_grid output takes the deposit
            # rule wherever a pixel is at least a cell.
            return max(sampling, (k - 0.5) * g - 0.5) if g >= 1 else sampling

        np.testing.assert_array_equal(supported, expected(deposit))
        assert not (supported & ~footprint_set).any()
        lost = int(footprint_set.sum() - supported.sum())
        lost = int(footprint_set.sum() - supported.sum())
        assert lost == _LOST_TO_DEPOSIT.get((reconstruction, pitch_lod), 0)
        _, wide = render(3)
        np.testing.assert_allclose(
            np.asarray(wide.star_leakage)[0][supported],
            leakage[supported],
            rtol=64 * np.finfo(float).eps,
            atol=0,
        )
    assert supported.any() and not supported.all()
    assert np.all(leakage[supported] > 0)


@pytest.mark.parametrize(
    "band_averaged", [True, False], ids=["band_center", "per_node"]
)
def test_stellar_and_planet_images_follow_product_scale(
    gaussian_observation, band_averaged
):
    """Compare node images and streamed counts to independent Gaussian integrals.

    The scene supplies the planet's sky position. Fluxes, gain and QE are
    analytic. SciPy and JAX evaluate the same erf integrals, so 1e-10 of each
    component peak allows only erf and floating-point summation error.
    """
    from conftest import gaussian_pixels

    import coronagraphoto as cp

    scene, path, exposure = gaussian_observation(band_averaged)
    q = exposure.quadrature
    assert len(q.pieces[0]) == 2
    band = q.bands[0]
    frame = path.coronagraph.frame
    assert band.center_nm == 0.9 * frame.reference_wavelength_nm
    waves = np.asarray(q.nodes_nm[0])
    east, north = np.asarray(scene.positions(np.array([0.0])))[:, 0, 0]

    def images(scale_nm):
        lod = float(frame.lod_to_arcsec(scale_nm))

        def gaussian(x, y, sigma):
            return gaussian_pixels(path.detector, frame, scale_nm, x, y, sigma)

        diameter = 0.013 / lod
        star = gaussian(-0.6, 0.45, np.hypot(0.28, diameter / 2))
        star += 0.6 * gaussian(0.8, -0.35, np.hypot(0.42, diameter / 2))
        planet = gaussian(-east / lod, north / lod, 0.35)
        return np.stack([1e6 * 1e-8 * star, 1e6 * 1e-7 * 0.3 * planet])

    photons = cp.photon_rates(scene, path, exposure)
    electrons = cp.expected(scene, path, exposure)
    assert np.all(photons.supported)
    gain = 20 * (0.4 + 0.004 * (waves - 500))
    eta = 0.9 - 0.002 * (waves - 500)
    truth = np.stack([images(band.center_nm if band_averaged else w) for w in waves])
    truth *= gain[:, None, None, None]
    integral = np.einsum("q,qcyx->cyx", q.weights()[0] * eta, truth)
    for k, field in enumerate(("star_leakage", "planet")):
        for got, want in (
            (getattr(photons, field)[0], truth[:, k]),
            (getattr(electrons, field)[0], integral[k]),
        ):
            np.testing.assert_allclose(got, want, atol=1e-10 * want.max(), rtol=0)
    if not band_averaged:
        fixed = images(band.center_nm) * np.sum(np.asarray(q.weights()[0]) * eta * gain)
        difference = np.max(np.abs(fixed - integral), axis=(1, 2)) / integral.max(
            axis=(1, 2)
        )
        # Center scaling changes the star by 1.40e-3 and the planet by 1.08e-3 of peak.
        assert np.all(difference > 1e-5)


class _BandScaledGaussian(IntegratedGaussianCoronagraph):
    """A build band sampled on a grid in reference-wavelength lambda/D."""

    def scale_wavelength_nm(self, wavelength_nm, band_center_nm):
        return band_center_nm * (
            self.frame.reference_wavelength_nm / self.bandpass.center_nm
        )


def _weighted_planet_sum(roll, scene, instrument, exposure):
    import equinox as eqx
    import jax.numpy as jnp

    import coronagraphoto as cp

    exposure = eqx.tree_at(lambda e: e.roll_deg, exposure, roll)
    image = cp.expected(scene, instrument, exposure).planet
    return jnp.sum(image * jnp.arange(image.shape[-1]))


@pytest.mark.parametrize("quantity", ["electrons", "photons", "roll_gradient"])
def test_band_planet_samples_match_per_node_reads(gaussian_observation, quantity):
    """Compare both planet readers and a roll derivative to per-node Gaussians.

    Each reference child converts the observing band's center to its native
    grid at every node. Reusing one read must give the same image and gradient.
    The floating-point tolerance (rtol 1e-13, atol 0) allows for roundoff
    in pixel integration and summation over nodes.
    """
    from dataclasses import replace

    import jax
    from optixstuff import MultiBandCoronagraph
    from optixstuff.spectral import Band

    import coronagraphoto as cp

    scene, instrument, _ = gaussian_observation()
    bands = (Band(520, 540), Band(560, 590))

    def setup(averaged):
        children = (
            _BandScaledGaussian(Band(550, 600), averaged, throughput=0.7),
            _BandScaledGaussian(Band(500, 550), averaged, throughput=0.1),
        )
        path = replace(instrument, coronagraph=MultiBandCoronagraph(children))
        exposure = cp.build_exposure(
            scene, path, bands, time_d=0.0, roll_deg=17.0, order=2, field_lod=0.0
        )
        return path, exposure

    path, exp = setup(True)
    reference, ref_exp = setup(False)
    np.testing.assert_array_equal(exp.quadrature.nodes_nm, ref_exp.quadrature.nodes_nm)
    np.testing.assert_array_equal(
        exp.quadrature.weights(), ref_exp.quadrature.weights()
    )
    if quantity == "roll_gradient":
        grad = jax.grad(_weighted_planet_sum)
        actual = grad(exp.roll_deg, scene, path, exp)
        truth = grad(ref_exp.roll_deg, scene, reference, ref_exp)
        assert abs(truth) > 0
    else:
        render = cp.expected if quantity == "electrons" else cp.photon_rates
        actual = render(scene, path, exp).planet
        truth = render(scene, reference, ref_exp).planet
    np.testing.assert_allclose(actual, truth, rtol=1e-13, atol=0)


def test_band_planet_psf_read_counts(gaussian_observation):
    """Count PSF calls and their sources with a callback in the actual read.

    The exact-algebra truth is one call with both planets per observing band,
    including under reverse differentiation. The per-node control reads both
    at every node and reads again when the checkpoint recomputes the node.
    With two nodes the measured forward and gradient counts were 1 and 1
    for the band average, and 2 and 4 for the per-node control.
    """
    import jax
    import jax.numpy as jnp
    from conftest import make_system
    from optixstuff.spectral import Band
    from skyscapes.scene import Planet

    import coronagraphoto as cp

    reads = []

    class CountedGaussian(IntegratedGaussianCoronagraph):
        def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
            # Tie the callback to the read's wavelength as well as its sources.
            jax.debug.callback(lambda x, _: reads.append(x.size), x_lod, wavelength_nm)
            return super().source_psfs(wavelength_nm, x_lod, y_lod, grid=grid)

    class CountedPerNode(_BandScaledGaussian, CountedGaussian):
        pass

    for model, averaged in ((CountedGaussian, True), (CountedPerNode, False)):
        coro = model(Band(500, 600), averaged)
        one, path, exp = gaussian_observation(coro=coro)
        scene = make_system(
            star=one.star,
            planet_groups=(Planet(jnp.ones(2), one.planets[0].physical_model),),
            separation_arcsec=jnp.array([0.02, 0.03]),
        )
        n = exp.quadrature.nodes_nm.shape[1]
        assert n > 1
        reads.clear()
        cp.expected(scene, path, exp)
        jax.effects_barrier()
        assert reads == [2] * (1 if averaged else n)

        reads.clear()
        jax.grad(_weighted_planet_sum)(exp.roll_deg, scene, path, exp)
        jax.effects_barrier()
        if averaged:
            assert reads == [2]
        else:
            assert len(reads) > n
            assert all(k == 2 for k in reads)


def test_build_exposure_refuses_underblurred_band(gaussian_observation):
    """The exact width ratio 1.3 exceeds 1.25, so only an explicit opt-in builds."""
    from optixstuff.spectral import Band

    import coronagraphoto as cp

    scene, instrument, _ = gaussian_observation()
    band = Band(444, 756)
    with pytest.raises(ValueError, match="nm"):
        cp.build_exposure(scene, instrument, (band,), time_d=0)
    exposure = cp.build_exposure(
        scene, instrument, (band,), time_d=0, allow_wide_bands=True
    )
    assert exposure.quadrature.bands == (band,)


def test_single_wavelength_product_changes_angular_image_at_two_nodes(
    gaussian_observation,
):
    """Two one-node exposures have different normalized stellar image shapes.

    Two separated lobes at fixed lambda/D offsets move on an angular detector.
    A 1e-5 peak difference is well above x64 erf and normalization roundoff.
    """
    from conftest import exposure_for
    from optixstuff.spectral import Quadrature

    import coronagraphoto as cp

    scene, instrument, exposure = gaussian_observation(False)
    q = exposure.quadrature
    images = []
    for j in (0, -1):
        one = Quadrature(q.bands, q.nodes_nm[:, [j]], np.ones((1, 1)))
        image = np.asarray(
            cp.expected(scene, instrument, exposure_for(scene, one)).star_leakage
        )
        images.append(image / image.max())
    assert np.max(np.abs(images[0] - images[1])) > 1e-5


def test_each_observing_band_uses_a_child_covering_both_edges(gaussian_observation):
    """Overlapping products give two band images with known Gaussian integrals.

    The first interval covers only the narrower observation. Each child's
    throughput multiplies a SciPy Gaussian integral at the observing centre
    and the node photon density. The 1e-10 peak tolerance allows erf and
    x64 roundoff.
    """
    from conftest import IntegratedGaussianCoronagraph, gaussian_pixels
    from optixstuff import MultiBandCoronagraph
    from optixstuff.spectral import Band

    import coronagraphoto as cp

    partial = IntegratedGaussianCoronagraph(Band(510, 545), True, throughput=0.1)
    wide = IntegratedGaussianCoronagraph(Band(515, 580), True, throughput=0.7)
    scene, path, _ = gaussian_observation(coro=MultiBandCoronagraph((partial, wide)))
    bands = (Band(520, 560), Band(524, 532))
    exposure = cp.build_exposure(scene, path, bands, time_d=0.0, order=2, field_lod=0.0)
    photons = cp.photon_rates(scene, path, exposure)
    east, north = np.asarray(scene.positions(np.array([0.0])))[:, 0, 0]
    for i, (band, throughput) in enumerate(zip(bands, (0.7, 0.1), strict=True)):
        lod = float(wide.frame.lod_to_arcsec(band.center_nm))
        pixels = gaussian_pixels(
            path.detector, wide.frame, band.center_nm, -east / lod, north / lod, 0.35
        )
        real = np.asarray(exposure.quadrature.active[i])
        waves = np.asarray(exposure.quadrature.nodes_nm[i])[real]
        gain = 20 * (0.4 + 0.004 * (waves - 500))
        density = (1e6 * 1e-7 * throughput * gain[:, None, None]) * pixels
        np.testing.assert_allclose(
            photons.planet[i, real], density, atol=1e-10 * density.max(), rtol=0
        )


def test_integer_nodes_keep_the_fractional_band_centre(gaussian_observation):
    """Integer and floating nodes of equal value give identical images.

    The band 520-561 nm has its centre at 540.5 nm; a band product read at a
    truncated 540 nm changes the stellar image by 1.3e-3 of its peak. Both
    exposures evaluate the same closed form, so 1e-12 of the peak allows only
    roundoff.
    """
    from conftest import exposure_for
    from optixstuff.spectral import Band, Quadrature

    import coronagraphoto as cp

    scene, instrument, _ = gaussian_observation()
    band = Band(520, 561)
    images = [
        np.asarray(
            cp.photon_rates(
                scene,
                instrument,
                exposure_for(scene, Quadrature((band,), nodes, [[20.5, 20.5]])),
            ).star_leakage
        )
        for nodes in (np.array([[530, 551]]), np.array([[530.0, 551.0]]))
    ]
    np.testing.assert_allclose(
        images[0], images[1], atol=1e-12 * images[1].max(), rtol=0
    )
