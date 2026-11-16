"""Positive time-dependent speckle excess in the stellar leakage budget."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from conftest import centroid, make_path, make_system
from hwoutils.epoch import Epoch
from hwoutils.grids import PixelGrid, SkyOrientation
from hwoutils.spectral import Band
from optixstuff import AbstractSpeckleField
from skyscapes.scene import Star

from coronagraphoto import image_rates


class Halo(AbstractSpeckleField):
    """A finite halo evolving in days, on its own instrumental grid."""

    pixel_scale_lod: float = 0.5
    epoch_d: float = 3.0

    def realize(self, *, wavelength_nm, time_d=0.0):
        """A time-dependent flux fraction per native pixel."""
        return (
            jnp.zeros((8, 8))
            .at[5, 6]
            .set(64e-10 * (0.5 + jnp.cos(time_d - self.epoch_d)))
        )


def test_speckle_leakage_day_clock():
    """Fails on epoch subtraction twice, wrong units or dead path wiring.

    Summed halo flux is conserved by covering coarsening grids: 64 eps.
    Roll must not rotate this instrumental contribution.
    """
    path = make_path()
    wave = jnp.array([400.0, 500.0, 600.0, 700.0])
    times = jnp.array([-10.0, 20.0])
    flux = jnp.array([[1.0, 3.0]] * 4)
    system = make_system(
        star=Star(wavelengths_nm=wave, times_d=times, flux_density_jy=flux)
    )
    bands = (Band(549.5, 550.5),)
    options = dict(quadrature_order=1)
    snapshots = []
    for time, factor in ((3.0, 1.5), (3.0 + np.pi, 0.0)):
        base = image_rates(
            system, path, bands, time_d=time, pointing=SkyOrientation(0), **options
        )
        timed = eqx.tree_at(
            lambda p: p.speckle, path, Halo(), is_leaf=lambda v: v is None
        )
        first = image_rates(
            system, timed, bands, time_d=time, pointing=SkyOrientation(0), **options
        )
        rolled = image_rates(
            system,
            path,
            bands,
            time_d=time,
            pointing=SkyOrientation(30),
            speckle=Halo(),
            **options,
        )
        snapshots.append(first)
        if factor > 0:
            grid = PixelGrid.regular(
                path.detector.shape, float(path.detector.pixel_scale_arcsec), "arcsec"
            )
            np.testing.assert_allclose(
                centroid(first.star_leakage[0, 0] - base.star_leakage[0, 0], grid),
                [grid.center_x + 5.0, grid.center_y + 3.0],
                rtol=0,
                atol=1e-10,  # Exact 2:1 sampling, floating-point subtraction.
            )
        phi = system.star.photon_flux_density(550.0, time, dist_pc=10.0)
        expected = 40 * 0.4 * phi * 64 * 1e-10 * factor
        np.testing.assert_allclose(
            (first.star_leakage - base.star_leakage).sum(),
            expected,
            rtol=64 * np.finfo(float).eps,
        )
        np.testing.assert_array_equal(first.star_leakage, rolled.star_leakage)

        # Same physical epoch: shift orbit, stellar samples and speckle clock.
        shift = 2.25
        shifted = make_system(
            star=Star(wavelengths_nm=wave, times_d=times - shift, flux_density_jy=flux),
            epoch=Epoch(2451547.0, 0.25),
            epoch_d=-shift,
        )
        same_date = image_rates(
            shifted,
            path,
            bands,
            time_d=time - shift,
            pointing=SkyOrientation(0),
            speckle=Halo(epoch_d=3.0 - shift),
            **options,
        )
        for a, b in zip(
            jax.tree.leaves(first), jax.tree.leaves(same_date), strict=True
        ):
            # Rebased cubic interpolation/orbits incur roundoff; 1e-10 relative.
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-18)

    vectorized = jax.vmap(
        lambda t: image_rates(
            system,
            path,
            bands,
            time_d=t,
            pointing=SkyOrientation(0),
            speckle=Halo(),
            **options,
        )
    )(jnp.array([3.0, 3.0 + np.pi]))
    for index, expected in enumerate(snapshots):
        for a, b in zip(
            jax.tree.leaves(vectorized), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(a[index], b, rtol=1e-10, atol=1e-18)
