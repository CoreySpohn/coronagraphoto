"""Sky offsets and disk geometry on even detector grids."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import GaussianCoronagraph, centroid, make_path, make_system
from hwoutils.grids import Frame, PixelGrid, SkyOrientation
from hwoutils.spectral import Band, Quadrature
from optixstuff import MultiBandCoronagraph, ScalarCoronagraph
from skyscapes.disk import ExovistaDisk
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import Planet

from coronagraphoto import image_rates


@pytest.mark.parametrize(
    "sky_pa,roll,expected",
    [(0.0, 90.0, (1.0, 0.0)), (90.0, 0.0, (-1.0, 0.0)), (30.0, 30.0, (0.0, 1.0))],
)
def test_cardinal_centroids(sky_pa, roll, expected):
    """Fails on parity/sign or half-pixel errors; interpolation bound .02 pixel."""
    path = make_path()
    scale = float(path.detector.pixel_scale_arcsec)
    system = make_system(sky_pa_deg=sky_pa, separation_arcsec=scale)
    # The smaller inscribed diameter enlarges lambda/D in the red band. A sky
    # source must still land on the same detector pixel.
    blue = GaussianCoronagraph(bandpass=Band(490, 510))
    red = GaussianCoronagraph(
        n=32, step=0.35, frame=Frame(600.0, 3.0, "inscribed"), bandpass=Band(590, 610)
    )
    multiband = MultiBandCoronagraph((blue, red))
    two_band_path = replace(
        path,
        primary=replace(
            path.primary, inscribed_diameter_m=3.0, collecting_area_m2=20.0
        ),
        coronagraph=multiband,
    )
    rates = image_rates(
        system,
        two_band_path,
        Quadrature.build(
            (Band(499.5, 500.5), Band(599.5, 600.5)), system, two_band_path, order=1
        ),
        time_d=0.0,
        pointing=SkyOrientation(roll),
    )
    for image in rates.planet[:, 0]:
        grid = PixelGrid.regular(path.detector.shape, scale, "arcsec")
        np.testing.assert_allclose(
            centroid(image, grid) - [grid.center_x, grid.center_y],
            expected,
            atol=0.02,
            rtol=0,
        )

    if sky_pa == 0.0:
        # Distinct fluxes and separations guard flattened body order and sums.
        wavelengths = jnp.array([400.0, 500.0, 600.0, 700.0])
        phases = jnp.array([0.0, 90.0, 180.0])
        contrasts = jnp.broadcast_to(
            jnp.array([1e-10, 2e-10])[:, None, None], (2, 4, 3)
        )
        combined = (
            Planet(jnp.ones(2), GridPhysicalModel(wavelengths, phases, contrasts)),
        )
        split = tuple(
            Planet(
                jnp.ones(1),
                GridPhysicalModel(wavelengths, phases, contrasts[k : k + 1]),
            )
            for k in range(2)
        )
        images = []
        for groups in (combined, split):
            scene = make_system(
                planet_groups=groups, separation_arcsec=jnp.array([scale, 2 * scale])
            )
            images.append(
                image_rates(
                    scene,
                    path,
                    Quadrature.build((Band(549.5, 550.5),), scene, path, order=1),
                    time_d=0.0,
                    pointing=SkyOrientation(90),
                ).planet[0, 0]
            )
        positions = scene.positions(jnp.array([0.0]))
        lod_arcsec = path.coronagraph.grid.frame.lod_to_arcsec(550.0)
        reference = path.coronagraph.source_psfs(
            550.0,
            positions[1, :, 0] / lod_arcsec,
            positions[0, :, 0] / lod_arcsec,
            grid=None,
        ).image.data
        reference = (
            40
            * 0.4
            * 1e6
            * (reference * jnp.array([1e-10, 2e-10])[:, None, None]).sum(0)
        )
        for image in images:
            # Same-grid interpolation and orbital propagation: roundoff only.
            np.testing.assert_allclose(image, reference, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("disk_kind", ["stored", "parametric"])
def test_roll_moves_disk_and_planet_together(disk_kind):
    """Fails if roll is dead for disks; major-axis alignment bound .02 radians.

    A stored inclined Gaussian ellipse has known north-south major axis;
    sampling of its second moments bounds rotation error below .02 radian.
    """
    path = make_path(n=32, step=0.3, cube=True)
    scale = float(path.detector.pixel_scale_arcsec)
    grid = PixelGrid.regular((32, 32), scale, "arcsec", orientation=SkyOrientation(0))
    x, y = grid.coordinates()
    contrast = 1e-9 * jnp.exp(-0.5 * ((x / (2 * scale)) ** 2 + (y / (6 * scale)) ** 2))
    disk = ExovistaDisk(
        grid, jnp.array([500.0, 600.0]), jnp.stack((contrast, contrast))
    )
    if disk_kind == "parametric":
        from skyscapes.disk import GraterDisk
        from zodi import DensityDust

        radius = 5 * scale * 10
        disk = GraterDisk(
            amplitude=DensityDust(
                level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
            ),
            sma_AU=radius,
            alpha_in=5.0,
            alpha_out=-5.0,
            ksi0_AU=0.02 * radius,
            gamma=2.0,
            beta=1.0,
            rmin_AU=0.1 * radius,
            rmax_AU=1.8 * radius,
            wavelengths_nm=jnp.array([500.0, 600.0]),
            g_HG_grid=jnp.zeros(2),
            Ag_grid=jnp.full(2, 0.5),
            n_slices_los=24,
        )
    system = make_system(separation_arcsec=5 * scale, disk=disk)
    system = eqx.tree_at(lambda s: s.midplane_inc_deg, system, 60.0)
    for roll in (0.0, 30.0):
        rates = image_rates(
            system,
            path,
            Quadrature.build((Band(549.5, 550.5),), system, path, order=1),
            time_d=0.0,
            pointing=SkyOrientation(roll),
        )
        delta = centroid(rates.planet[0, 0], grid) - [grid.center_x, grid.center_y]
        data = np.asarray(rates.exozodi[0, 0])
        moment = np.array(
            [
                [(data * x * x).sum(), (data * x * y).sum()],
                [(data * x * y).sum(), (data * y * y).sum()],
            ]
        )
        major = np.linalg.eigh(moment)[1][:, -1]
        assert (
            abs((major[0] * delta[1] - major[1] * delta[0]) / np.linalg.norm(delta))
            < 0.02
        )
        np.testing.assert_allclose(
            delta,
            5 * np.array([np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll))]),
            atol=0.02,
        )


def test_compact_disk_planet_centroids():
    """Fails on an even-grid half-pixel disk shift; bound .02 pixel.

    ExovistaDisk can store a compact one-pixel clump, so no profile skip is
    needed. At native sampling no source-grid interpolation error is allowed.
    """
    path = make_path(n=24, step=0.3, cube=True)
    grid = PixelGrid.regular(
        (24, 24),
        float(path.detector.pixel_scale_arcsec),
        "arcsec",
        orientation=SkyOrientation(0),
    )
    x, y = grid.pixel_to_unit(14.0, 15.0)
    east, north = grid.orientation.detector_to_sky(x, y)
    contrast = jnp.zeros((24, 24)).at[15, 14].set(1e-10)
    disk = ExovistaDisk(
        grid, jnp.array([500.0, 600.0]), jnp.stack((contrast, contrast))
    )
    system = make_system(
        separation_arcsec=float(jnp.hypot(east, north)),
        sky_pa_deg=float(jnp.rad2deg(jnp.arctan2(east, north))),
        disk=disk,
    )
    rates = image_rates(
        system,
        path,
        Quadrature.build((Band(549.5, 550.5),), system, path, order=1),
        time_d=0.0,
        pointing=SkyOrientation(0),
    )
    np.testing.assert_allclose(
        centroid(rates.planet[0, 0], grid),
        centroid(rates.exozodi[0, 0], grid),
        atol=0.02,
        rtol=0,
    )


@pytest.mark.parametrize("throughput_only", [False, True])
def test_imaging_capability_required(throughput_only):
    """ETC-only and throughput-only paths cannot produce detector pixels."""
    path = make_path(n=8)
    scalar = ScalarCoronagraph(
        grid=path.coronagraph.grid,
        throughput_fraction=0.3,
        aperture_area_lod2=1.0,
        stellar_leakage_density_lod2=1e-9,
    )
    path = eqx.tree_at(
        lambda p: p.coronagraph, path, None if throughput_only else scalar
    )
    with pytest.raises(TypeError, match="AbstractImagingCoronagraph"):
        image_rates(
            make_system(),
            path,
            Quadrature.monochromatic(550.0),
            time_d=0.0,
            pointing=SkyOrientation(0),
        )
