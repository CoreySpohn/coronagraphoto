"""YIP adapter integration using a small synthetic FITS package."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    make_path,
    make_system,
    with_zodi,
    write_yip,
)
from hwoutils.constants import arcsec2rad
from optixstuff import Exposure
from optixstuff.grids import PixelGrid, SkyOrientation, resample
from optixstuff.spectral import Band, Quadrature
from skyscapes import PrecomputedZodi
from skyscapes.scene import BlackbodyStar, FlatStar
from yippy import open_yip

from coronagraphoto import photon_rates


@pytest.mark.parametrize("distinct_products", [False, True])
def test_yip_image_absolute_photon_fraction(tmp_path, distinct_products):
    """Fails if the YIP centre density is summed without its native cell area.

    Gaussian omitted tails <2e-8 at this offset; float32 YIP interpolation
    adds <2e-6 relative. A pixel-area error changes the result by a factor 4.
    Stellar and transmission maps end at their tabulated footprint: beyond it
    the independent oracle is zero (``outside == "zero"``), and the reader
    gives exactly zero with ``supported`` False for every pixel the
    reconstruction cannot reach. The distinct map's smooth positive gradient
    also stays positive here. Bound the flux added by clipping negative
    leakage to 64 eps of total absolute stellar flux on the supported pixels;
    the nonnegative-rate guard must not hide finite-field ringing.
    """
    coro = open_yip(
        write_yip(tmp_path / "yip", distinct_products=distinct_products)
    ).prepare()
    if distinct_products:
        # A YIP has one FITS grid for the stellar/off-axis pair. Give the
        # in-memory products distinct frames to exercise the reader contract.
        stellar_grid = coro.stellar_grid
        stellar_frame = replace(
            stellar_grid.frame, diameter_m=3.0, reference_wavelength_nm=650.0
        )
        coro = eqx.tree_at(
            lambda c: c.stellar_grid, coro, replace(stellar_grid, frame=stellar_frame)
        )
    path = make_path(n=24, step=0.5, detector_n=28, detector_step=0.5)
    path = eqx.tree_at(lambda p: p.coronagraph, path, coro)
    # Exactly a measured source offset: 1 lod north at the reference wavelength.
    stellar = coro.stellar_map(550.0, 0.1, grid=None)
    star = FlatStar(
        flux_phot_per_nm_m2=1e6,
        fixed_diameter_arcsec=0.1 * stellar.grid.frame.lod_to_arcsec(550.0),
    )
    if distinct_products:
        star = BlackbodyStar(teff_K=5772.0, radius_m=6.957e8)
        stellar = coro.stellar_map(
            550.0,
            star.angular_diameter_arcsec(10.0)
            / stellar.grid.frame.lod_to_arcsec(550.0),
            grid=None,
        )
    phi_star = star.photon_flux_density(550.0, 0.0, dist_pc=10.0)
    system = make_system(
        star=star, separation_arcsec=float(coro.grid.frame.lod_to_arcsec(550.0))
    )
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    quadrature = Quadrature.build((Band(549.5, 550.5),), system, path, zodi, order=1)
    rates = photon_rates(
        with_zodi(system, zodi),
        path,
        Exposure(time_d=0.0, roll_deg=90, quadrature=quadrature),
    )
    np.testing.assert_allclose(
        rates.planet.sum(), 40 * 0.4 * phi_star * 1e-10 * 0.3, rtol=2e-6
    )

    # Use native products directly as an independent geometry oracle. A finite
    # diameter and spatial gradients expose both diameter and frame conversion.
    grid = PixelGrid.regular(
        (28, 28),
        float(path.detector.pixel_scale_arcsec),
        "arcsec",
        orientation=SkyOrientation(90),
    )
    for product, term, scale in (
        (stellar.integrated(), rates.star_leakage[0, 0], 40 * 0.4 * phi_star),
        (
            coro.sky_trans,
            rates.local_zodi[0, 0],
            40 * 0.4 * 1e12 * grid.pixel_area * arcsec2rad**2,
        ),
    ):
        angular = replace(product.grid.to_arcsec(550.0), orientation=grid.orientation)
        assert product.outside == "zero"
        expected = resample(replace(product, grid=angular), grid).data * scale
        covered = np.asarray(rates.supported[0])
        assert covered.any()
        np.testing.assert_array_equal(np.asarray(term)[~covered], 0.0)
        expected = jnp.where(covered, expected, 0.0)
        negative_flux = float(jnp.maximum(-expected, 0).sum())
        print(
            "map",
            product.quantity,
            "minimum",
            float(expected.min()),
            "negative flux",
            negative_flux,
        )
        assert negative_flux <= 64 * np.finfo(float).eps * float(
            jnp.abs(expected).sum()
        )
        np.testing.assert_allclose(term, expected, rtol=2e-6, atol=1e-16)
