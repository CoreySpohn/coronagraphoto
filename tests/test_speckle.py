"""Signed time-dependent speckle excess in the stellar leakage budget."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from conftest import centroid, make_path, make_system
from hwoutils.epoch import Epoch
from hwoutils.grids import Frame, Image, Measure, PixelGrid
from hwoutils.spectral import Band, Quadrature
from optixstuff import AbstractSpeckleField, Exposure
from skyscapes import Scene
from skyscapes.scene import Star

from coronagraphoto import photon_rates


class Halo(AbstractSpeckleField):
    """A finite halo evolving in days, on its own instrumental grid."""

    grid: PixelGrid = eqx.field(
        static=True,
        default=PixelGrid.focal(8, 0.5, Frame(550.0, 6.0, "circumscribed")),
    )
    epoch_d: float = 3.0
    level: float = 1.5

    def spectral_breaks_nm(self):
        """The halo is achromatic."""
        return np.empty(0)

    def realize(self, *, wavelength_nm, time_d=0.0):
        """A time-dependent flux fraction per native pixel."""
        data = (
            jnp.zeros((8, 8))
            .at[5, 6]
            .set(64e-10 * (self.level + jnp.cos(time_d - self.epoch_d)))
        )
        return Image(
            data,
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
            outside="reflect",
        )


def test_speckle_leakage_day_clock():
    """Fails on epoch subtraction twice, wrong units or dead path wiring.

    Over exactly one reflected halo footprint the linear reconstruction
    conserves its native flux at 2:1 sampling, to 64 eps. A larger detector
    would include reflected replicas. Roll cannot rotate this instrumental halo.
    """
    path = make_path(detector_n=16)
    wave = jnp.array([400.0, 500.0, 600.0, 700.0])
    times = jnp.array([-10.0, 20.0])
    flux = jnp.array([[1.0, 3.0]] * 4)
    system = make_system(
        star=Star(wavelengths_nm=wave, times_d=times, flux_density_jy=flux)
    )
    q = Quadrature.build((Band(549.5, 550.5),), system, path, Halo(), order=1)
    scene = Scene(system)

    def rates(scene, path, time_d, roll_deg=0, speckle=None):
        if speckle is not None:
            path = eqx.tree_at(
                lambda p: p.speckle, path, speckle, is_leaf=lambda v: v is None
            )
        return photon_rates(
            scene, path, Exposure(time_d=time_d, roll_deg=roll_deg, quadrature=q)
        )

    snapshots = []
    for time, factor in ((3.0, 2.5), (3.0 + np.pi, 0.5)):
        base = rates(scene, path, time)
        first = rates(scene, path, time, speckle=Halo())
        rolled = rates(scene, path, time, roll_deg=30, speckle=Halo())
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
        same_date = rates(
            Scene(shifted), path, time - shift, speckle=Halo(epoch_d=3.0 - shift)
        )
        for a, b in zip(
            jax.tree.leaves(first), jax.tree.leaves(same_date), strict=True
        ):
            # Rebased cubic interpolation/orbits incur roundoff; 1e-10 relative.
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-18)

    # A signed excess lowers the floor without clipping the total: -0.01 of
    # the halo spreads to at most ~2e-11 per detector pixel, under the 6.25e-11
    # fixture floor. Fails if the excess itself is clipped at zero. The sum
    # cancels against a floor ~250 times larger: 1e-10 relative.
    base = rates(scene, path, 3.0)
    dimmed = rates(scene, path, 3.0, speckle=Halo(level=-1.01))
    phi = system.star.photon_flux_density(550.0, 3.0, dist_pc=10.0)
    np.testing.assert_allclose(
        (dimmed.star_leakage - base.star_leakage).sum(),
        40 * 0.4 * phi * 64e-10 * -0.01,
        rtol=1e-10,
    )

    vectorized = jax.vmap(lambda t: rates(scene, path, t, speckle=Halo()))(
        jnp.array([3.0, 3.0 + np.pi])
    )
    for index, expected in enumerate(snapshots):
        for a, b in zip(
            jax.tree.leaves(vectorized), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(a[index], b, rtol=1e-10, atol=1e-18)
