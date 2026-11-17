"""YIP adapter integration using a small synthetic FITS package."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_path, make_system, write_yip
from hwoutils.constants import arcsec2rad
from hwoutils.grids import PixelGrid, SkyOrientation, resample
from hwoutils.spectral import Band
from optixstuff import YippyCoronagraph
from skyscapes.scene import BlackbodyStar, FlatStar
from yippy import Coronagraph, EqxCoronagraph
from zodi.jax import PrecomputedZodi

from coronagraphoto import image_rates


@pytest.mark.parametrize("distinct_products", [False, True])
def test_yip_image_absolute_photon_fraction(tmp_path, distinct_products):
    """Fails if a YIP per-pixel fraction is interpreted as a density.

    Gaussian omitted tails <2e-8 at this offset; float32 YIP interpolation
    adds <2e-6 relative. A pixel-area error changes the result by a factor 4.
    Stellar and transmission maps use the even extension, so a constant
    stellar map remains constant beyond its footprint. The distinct map's
    smooth positive gradient also stays positive here. Bound the flux added
    by clipping negative leakage to 64 eps of total absolute stellar flux;
    the nonnegative-rate guard must not hide finite-field ringing.
    """
    host = Coronagraph(write_yip(tmp_path / "yip", distinct_products=distinct_products))
    if distinct_products:
        # The legacy loader assigns the stellar header to the off-axis product.
        # Restore its independent synthetic geometry before building the adapter.
        host.offax.grid = make_path(n=24, step=0.5).coronagraph.grid
        host.grid = host.offax.grid
    coro = YippyCoronagraph(
        backend=EqxCoronagraph(yippy_coro=host, ensure_psf_datacube=False)
    )
    path = make_path(n=24, step=0.5, detector_n=28, detector_step=0.5)
    path = eqx.tree_at(lambda p: p.coronagraph, path, coro)
    # Exactly a measured source offset: 1 lod north at the reference wavelength.
    stellar = coro.stellar_intens(0.1)
    star = FlatStar(
        flux_phot_per_nm_m2=1e6,
        diameter_arcsec=0.1 * stellar.grid.frame.lod_to_arcsec(550.0),
    )
    if distinct_products:
        star = BlackbodyStar(teff_K=5772.0, radius_m=6.957e8)
        stellar = coro.stellar_intens(
            star.diameter_arcsec(dist_pc=10.0) / stellar.grid.frame.lod_to_arcsec(550.0)
        )
    phi_star = star.photon_flux_density(550.0, 0.0, dist_pc=10.0)
    system = make_system(
        star=star, separation_arcsec=float(coro.grid.frame.lod_to_arcsec(550.0))
    )
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e12))
    rates = image_rates(
        system,
        path,
        (Band(549.5, 550.5),),
        time_d=0.0,
        pointing=SkyOrientation(90),
        quadrature_order=1,
        zodi=zodi,
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
        (stellar, rates.star_leakage[0, 0], 40 * 0.4 * phi_star),
        (
            coro.sky_trans,
            rates.local_zodi[0, 0],
            40 * 0.4 * 1e12 * grid.pixel_area * arcsec2rad**2,
        ),
    ):
        angular = replace(product.grid.to_arcsec(550.0), orientation=grid.orientation)
        assert product.outside == "reflect"
        expected = resample(replace(product, grid=angular), grid).data * scale
        negative_flux = float(jnp.maximum(-expected, 0).sum())
        print(
            "reflected map",
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
