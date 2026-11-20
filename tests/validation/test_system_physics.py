"""Calculator-to-image aperture and point-to-extended-source closure."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    GaussianSceneCoronagraph,
    PlateauCoronagraph,
    make_path,
    make_system,
    with_zodi,
)
from hwoutils.constants import arcsec2rad
from optixstuff import Exposure
from optixstuff.apertures import Circular
from optixstuff.grids import Image, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Quadrature
from skyscapes import PrecomputedZodi
from skyscapes.disk import ExovistaDisk

from coronagraphoto import photon_rates


def test_aperture_closure_two_samplings():
    """Fails on area/unit/throughput errors in any of the four image terms.

    Planet/leakage/zodi retain exact plateau integrals. A finite uniform
    disk convolved with the square PSF is a separable tapered map, not a
    global plateau. Along each native axis h[j] is the sum of the square's
    unit interior and half boundary weights over source pixels, weighted
    by their deposited flux. Linear sky sampling and native deposition give
    that source vector as (detector_step/native_step)*W.T@W@ones, where W
    is the triangular interpolation kernel; at 2:1 its edge entries are
    25/32, not one. The 2-D map is f*.3/16*step**2*h[y]*h[x].
    Zero-extended band-limited pixel
    integration multiplies each Fourier mode by sinc(frequency*pixel_width).
    The independently integrated separable DFT supplies the disk oracle;
    256 eps relative allows FFT, contraction and weighted-sum rounding.
    """
    coro = PlateauCoronagraph(n=32, step=0.25)
    lod_arcsec = float(coro.grid.frame.lod_to_arcsec(550.0))
    native_sky = replace(coro.grid.to_arcsec(550.0), orientation=SkyOrientation(0))
    disk = ExovistaDisk(
        native_sky, jnp.array([500.0, 600.0]), jnp.full((2, 32, 32), 1e-10)
    )
    system = make_system(
        separation_arcsec=0.5 * lod_arcsec, sky_pa_deg=270.0, disk=disk
    )
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    rule = Circular(0.7)
    record = coro.curves(rule)
    phi_p = float(system.planet_flux_densities(550.0, jnp.array([0.0]))[0, 0])
    phi_s = float(system.star.photon_flux_density(550.0, 0.0, dist_pc=10.0))
    lz = float(zodi.photon_radiance(550.0, system.sky_frame, 0.0))
    ld = float(system.photon_radiance(550.0, 0.0, grid=native_sky).data[16, 16])
    omega = (
        float(record.evaluate("core_area_lod2", 0.5)) * lod_arcsec**2 * arcsec2rad**2
    )
    transmission = float(coro.uniform_sky_response(0.5, 550.0))
    reference = (
        40
        * 0.4
        * np.array(
            [
                float(record.evaluate("throughput", 0.5)) * phi_p,
                float(record.evaluate("leakage_fraction", 0.5)) * phi_s,
                transmission * lz * omega,
                transmission * ld * omega,
            ]
        )
    )
    measured, references = [], []
    for n, step in ((32, 0.25), (64, 0.125)):
        path = make_path(n=32, step=0.25, detector_n=n, detector_step=step)
        path = eqx.tree_at(lambda p: p.coronagraph, path, coro)
        quadrature = Quadrature.build(
            (Band(549.5, 550.5),), system, path, zodi, order=1
        )
        rates = photon_rates(
            with_zodi(system, zodi),
            path,
            Exposure(time_d=0.0, roll_deg=0, quadrature=quadrature),
        )
        aperture_grid = PixelGrid.regular((n, n), step, "lod", frame=coro.grid.frame)
        psf = Image(
            rates.planet[0, 0],
            aperture_grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )
        weights = rule.weights(psf, (0.5, 0.0))
        # Independent finite-source convolution and exact Fourier pixel integrals.
        delta = (np.arange(32)[:, None] - np.arange(32)[None, :]) * 0.25
        sky_positions = (np.arange(n) - (n - 1) / 2) * step / 0.25 + 15.5
        linear = np.maximum(1 - abs(sky_positions[:, None] - np.arange(32)), 0)
        source = step / 0.25 * linear.T @ linear @ np.ones(32)
        h = ((abs(delta) < 2) + 0.5 * (abs(delta) == 2)) @ source
        padded = np.pad(h, (16, 16))
        frequencies = np.fft.fftfreq(len(padded))
        target = (np.arange(n) - (n - 1) / 2) * step / 0.25 + 15.5 + 16
        basis = np.exp(2j * np.pi * target[:, None] * frequencies)
        integrated = (
            basis @ (np.fft.fft(padded) * np.sinc(frequencies * step / 0.25))
        ).real / len(padded)
        disk_pixels = (
            40
            * 0.4
            * phi_s
            * 1e-10
            * 0.3
            / 16
            * step**2
            * integrated[:, None]
            * integrated[None, :]
        )
        sample_reference = reference.copy()
        sample_reference[3] = np.sum(disk_pixels * np.asarray(weights))
        references.append(sample_reference)
        measured.append(
            np.array(
                [
                    float((getattr(rates, name)[0, 0] * weights).sum())
                    for name in ("planet", "star_leakage", "local_zodi", "exozodi")
                ]
            )
        )
    gap = abs(measured[1] - measured[0])
    print(
        "closure reference",
        reference,
        "coarse",
        measured[0],
        "fine",
        measured[1],
        "gap",
        gap,
    )
    for sample, sample_reference in zip(measured, references, strict=True):
        np.testing.assert_allclose(
            sample, sample_reference, rtol=256 * np.finfo(float).eps, atol=0
        )


def test_disk_equals_pixel_center_source():
    """Fails on an even-grid half-pixel offset or a disk without a scene operator.

    A one-pixel scene contracts to its point PSF before reconstruction.
    Both outputs then integrate the same zero-extended band-limited density
    over detector pixels, so equality holds to 64 eps of peak. Declaring the
    point PSF linear would compare different continuous images. A source
    shift of half a native pixel differs by >.05 of peak.
    """

    class BandLimitedGaussian(GaussianSceneCoronagraph):
        def _native_source_psfs(self, x_lod, y_lod):
            psfs = super()._native_source_psfs(x_lod, y_lod)
            return psfs._replace(
                image=replace(psfs.image, reconstruction="band_limited")
            )

    path = make_path(n=20, step=0.3)
    path = eqx.tree_at(
        lambda p: p.coronagraph, path, BandLimitedGaussian(n=20, step=0.3)
    )
    grid = PixelGrid.regular(
        (20, 20),
        float(path.detector.pixel_scale_arcsec),
        "arcsec",
        orientation=SkyOrientation(0),
    )
    x, y = grid.pixel_to_unit(12.0, 13.0)
    east, north = grid.orientation.detector_to_sky(x, y)
    flux_fraction = 1e-10
    delta = jnp.zeros(grid.shape).at[13, 12].set(flux_fraction)
    disk = ExovistaDisk(grid, jnp.array([500.0, 600.0]), jnp.stack((delta, delta)))
    system = make_system(
        contrast=flux_fraction,
        separation_arcsec=float(jnp.hypot(x, y)),
        sky_pa_deg=float(jnp.rad2deg(jnp.arctan2(east, north))),
        disk=disk,
    )
    scene = system
    exposure = Exposure(
        time_d=0.0,
        roll_deg=0,
        quadrature=Quadrature.build((Band(549.5, 550.5),), system, path, order=1),
    )
    rates = photon_rates(scene, path, exposure)
    a, b = np.asarray(rates.planet), np.asarray(rates.exozodi)
    np.testing.assert_allclose(a, b, atol=64 * np.finfo(float).eps * a.max(), rtol=0)
    native = path.coronagraph.grid
    shifted = path.coronagraph.source_psfs(
        550.0,
        jnp.array([float(x) / float(native.frame.lod_to_arcsec(550.0)) + 0.15]),
        jnp.array([float(y) / float(native.frame.lod_to_arcsec(550.0))]),
        grid=None,
    ).image.data
    correct = a[0] / (40 * 0.4 * 1e6 * flux_fraction)
    assert np.max(abs(shifted - correct)) / correct.max() > 0.05
    missing = make_path(n=20, step=0.3)
    with pytest.raises(TypeError, match="AbstractSceneCoronagraph"):
        photon_rates(scene, missing, exposure)
