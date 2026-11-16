"""Calculator-to-image aperture and point-to-extended-source closure."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PlateauCoronagraph, make_path, make_system
from hwoutils.constants import arcsec2rad
from hwoutils.grids import Image, PixelGrid, SkyOrientation
from hwoutils.spectral import Band
from skyscapes.background import PrecomputedZodi
from skyscapes.disk import ExovistaDisk
from yippy import Circular

from coronagraphoto import image_rates


def test_aperture_closure_two_samplings():
    """Fails on area/unit/throughput errors in any of the four image terms.

    Exact plateau integrals isolate radiometry. Each sampling must agree
    with the record within half the two-sampling absolute gap plus 1e-9
    photons/s/nm. This is closure, not a convergence-order claim.
    """
    coro = PlateauCoronagraph(n=32, step=0.25, cube=True)
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
    lz = float(
        zodi.photon_radiance(
            550.0, ecliptic_latitude_deg=0.0, solar_longitude_deg=135.0
        )
    )
    ld = float(system.photon_radiance(550.0, 0.0, grid=native_sky).data[16, 16])
    omega = (
        float(record.evaluate("core_area_lod2", 0.5)) * lod_arcsec**2 * arcsec2rad**2
    )
    reference = (
        40
        * 0.4
        * np.array(
            [
                float(record.evaluate("throughput", 0.5)) * phi_p,
                float(record.evaluate("leakage_fraction", 0.5)) * phi_s,
                float(record.evaluate("occulter_transmission", 0.5)) * lz * omega,
                float(record.evaluate("occulter_transmission", 0.5)) * ld * omega,
            ]
        )
    )
    measured = []
    for n, step in ((32, 0.25), (64, 0.125)):
        path = make_path(n=32, step=0.25, detector_n=n, detector_step=step)
        path = eqx.tree_at(lambda p: p.coronagraph, path, coro)
        rates = image_rates(
            system,
            path,
            (Band(549.5, 550.5),),
            time_d=0.0,
            pointing=SkyOrientation(0),
            quadrature_order=1,
            zodi=zodi,
        )
        aperture_grid = PixelGrid.regular((n, n), step, "lod", frame=coro.grid.frame)
        psf = Image(
            rates.planet[0, 0], aperture_grid, (), "per_pixel", "stellar_flux_fraction"
        )
        weights = rule.weights(psf, (0.5, 0.0))
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
    for sample in measured:
        np.testing.assert_array_less(abs(sample - reference), gap / 2 + 1e-9)


def test_extended_scene_equals_pixel_center_source():
    """Fails on an even-grid half-pixel offset or silent missing-cube fallback.

    Native-grid delta contraction equals direct Gaussian at 64 eps of peak;
    a half-native-pixel source shift differs by >.05 of peak.
    """
    path = make_path(n=20, step=0.3, cube=True)
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
    options = dict(time_d=0.0, pointing=SkyOrientation(0), quadrature_order=1)
    bands = (Band(549.5, 550.5),)
    rates = image_rates(system, path, bands, **options)
    a, b = np.asarray(rates.planet), np.asarray(rates.exozodi)
    np.testing.assert_allclose(a, b, atol=64 * np.finfo(float).eps * a.max(), rtol=0)
    native = path.coronagraph.grid
    shifted = path.coronagraph.create_psfs(
        jnp.array([float(x) / float(native.frame.lod_to_arcsec(550.0)) + 0.15]),
        jnp.array([float(y) / float(native.frame.lod_to_arcsec(550.0))]),
    ).data
    correct = a[0] / (40 * 0.4 * 1e6 * flux_fraction)
    assert np.max(abs(shifted - correct)) / correct.max() > 0.05
    missing = eqx.tree_at(lambda p: p.coronagraph.psf_datacube, path, None)
    with pytest.raises(ValueError, match="psf_datacube"):
        image_rates(system, missing, bands, **options)
