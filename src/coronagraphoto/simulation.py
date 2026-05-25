"""Functions for running full simulations and processing sources.

Public API conventions:

- ``gen_<source>_count_rate(source, optical_path, *, ...)`` returns the
  noiseless per-pixel photo-electron rate on the detector for one source.
- ``sim_<source>(source, optical_path, prng_key, *, ...)`` returns a
  noisy detector readout (photon Poisson + QE binomial) for one source.
- ``system_readout(scene, optical_path, prng_key, *, ...)`` sums every
  source in the scene into a single readout.

All observation parameters (``start_time_jd``, ``exposure_time_s``,
``wavelength_nm``, ``bin_width_nm``, ``telescope_pa_deg``,
``ecliptic_lat_deg``, ``solar_lon_deg``) are kwarg-only. The convention
keeps signatures discoverable when more parameters land later (IFS,
multi-roll observations).
"""

import jax
import jax.numpy as jnp
from hwoutils.conversions import arcsec_to_lambda_d, lambda_d_to_arcsec
from hwoutils.transforms import ccw_rotation_matrix, resample_flux
from skyscapes.background import ZodiSource


def pre_coro_bin_processing(flux, bin_center_nm, bin_width_nm, optical_path):
    """Process a bin through the pre-coro elements of the optical path."""
    # ph/s/m^2/nm -> ph/s/m^2
    flux = flux * bin_width_nm
    # ph/s
    flux = flux * optical_path.primary.area_m2
    # apply combined attenuation of mirrors / filters / etc.
    return flux * optical_path.system_throughput(bin_center_nm)


def _resample_to_detector(image_rate_coro, bin_center_nm, optical_path):
    """Resample a coronagraph-plane image onto the detector pixel grid.

    Pipeline geometry, not detector hardware: needs the coronagraph's
    plate scale (lambda/D / px), the detector's plate scale (arcsec/px),
    the wavelength, and the primary diameter to convert lambda/D to
    arcsec.
    """
    inc_pixel_scale_arcsec = lambda_d_to_arcsec(
        optical_path.coronagraph.pixel_scale_lod,
        bin_center_nm,
        optical_path.primary.diameter_m,
    )
    return resample_flux(
        image_rate_coro,
        inc_pixel_scale_arcsec,
        optical_path.detector.pixel_scale,
        optical_path.detector.shape,
        0.0,  # rotation is applied source-side, not detector-side
    )


def post_coro_bin_processing(image_rate_coro, bin_center_nm, optical_path):
    """Process a bin through the post-coro elements of the optical path."""
    image_rate_detector = _resample_to_detector(
        image_rate_coro, bin_center_nm, optical_path
    )
    return jnp.clip(image_rate_detector, 0, None)


# ---------------------------------------------------------------------------
# Star
# ---------------------------------------------------------------------------


def star_rate(
    star,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
):
    """Generate the star count rate on the detector."""
    source_diam_lod = arcsec_to_lambda_d(
        star.diameter_arcsec, wavelength_nm, optical_path.primary.diameter_m
    )
    flux = star.spec_flux_density(wavelength_nm, start_time_jd)
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)
    image_rate_coro = optical_path.coronagraph.stellar_intens(source_diam_lod) * flux
    return post_coro_bin_processing(image_rate_coro, wavelength_nm, optical_path)


def star_readout(
    star,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
):
    """Process a star through the provided optical path."""
    image_rate_detector = star_rate(
        star,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------


def planet_rate(
    planet,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    trig_solver,
):
    """Generate the per-batch planet count rate on the detector.

    Operates on a single ``skyscapes.scene.Planet`` (which internally
    batches K planets sharing the same atmosphere class). The Python
    loop over a heterogeneous ``System.planets`` tuple lives in
    :func:`system_readout`; this function stays inside the per-Planet-type
    JIT cache boundary (see ``brain/Planet Loop Architecture.md``).
    """
    # The new Planet API takes a 1-D time axis; squeeze T=1.
    source_positions_as = planet.position_arcsec(
        trig_solver, jnp.atleast_1d(start_time_jd), star=star
    )[:, :, 0]  # (2, K)

    # A positive telescope PA corresponds to a CW rotation of the sky.
    rotation_matrix = ccw_rotation_matrix(-telescope_pa_deg)
    source_positions_as = rotation_matrix @ source_positions_as

    source_positions_lod = arcsec_to_lambda_d(
        source_positions_as, wavelength_nm, optical_path.primary.diameter_m
    )

    # ``wavelength_nm`` stays scalar -- the underlying atmosphere reflectivity
    # code expects a scalar and broadcasts internally. ``start_time_jd`` is
    # promoted to (1,) because the orbit propagator needs a T axis.
    flux = planet.spec_flux_density(
        trig_solver,
        wavelength_nm,
        jnp.atleast_1d(start_time_jd),
        star=star,
    )[:, 0]  # (K,) -- drop T=1 axis
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    psfs = optical_path.coronagraph.create_psfs(
        source_positions_lod[0], source_positions_lod[1]
    )
    image_rate_coro = jnp.einsum("i,ijk->jk", flux, psfs)
    return post_coro_bin_processing(image_rate_coro, wavelength_nm, optical_path)


def planet_readout(
    planet,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    trig_solver,
):
    """Process a per-batch Planet through the optical path."""
    image_rate_detector = planet_rate(
        planet,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        telescope_pa_deg=telescope_pa_deg,
        star=star,
        trig_solver=trig_solver,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------


def _convolve_quadrants(flux, psf_datacube):
    """Convolve flux with a quarter-symmetric PSF datacube via fold-and-sum.

    Handles padding dynamically to ensure all quadrants match the shape of the
    first quadrant (which defines the PSF datacube shape).
    """
    ny, nx = flux.shape
    cy, cx = (ny - 1) // 2, (nx - 1) // 2

    # Q1: top-right (includes center pixel and axes) -> reference shape
    q1 = flux[cy:, cx:]
    target_h, target_w = q1.shape

    # Q2: top-left -- flip X, pad inner-left + outer-right to target width
    q2_raw = flux[cy:, :cx]
    q2_flipped = q2_raw[:, ::-1]
    pad_q2_right = max(0, target_w - (q2_flipped.shape[1] + 1))
    q2 = jnp.pad(q2_flipped, ((0, 0), (1, pad_q2_right)))

    # Q3: bottom-left -- flip both, pad inner-top, inner-left, outer-bottom/right
    q3_raw = flux[:cy, :cx]
    q3_flipped = q3_raw[::-1, ::-1]
    pad_q3_bottom = max(0, target_h - (q3_flipped.shape[0] + 1))
    pad_q3_right = max(0, target_w - (q3_flipped.shape[1] + 1))
    q3 = jnp.pad(q3_flipped, ((1, pad_q3_bottom), (1, pad_q3_right)))

    # Q4: bottom-right -- flip Y, pad inner-top + outer-bottom
    q4_raw = flux[:cy, cx:]
    q4_flipped = q4_raw[::-1, :]
    pad_q4_bottom = max(0, target_h - (q4_flipped.shape[0] + 1))
    q4 = jnp.pad(q4_flipped, ((1, pad_q4_bottom), (0, 0)))

    flux_stack = jnp.stack([q1, q2, q3, q4])
    partial_images = jnp.einsum("qij,ijxy->qxy", flux_stack, psf_datacube)

    img_q1 = partial_images[0]
    img_q2 = jnp.fliplr(partial_images[1])
    img_q3 = jnp.flipud(jnp.fliplr(partial_images[2]))
    img_q4 = jnp.flipud(partial_images[3])

    return img_q1 + img_q2 + img_q3 + img_q4


def disk_rate(
    disk,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    incl_deg,
    pa_deg,
):
    """Generate the disk count rate on the detector.

    Disks return CONTRAST (dimensionless flux ratio relative to the host
    star); we multiply by ``star.spec_flux_density`` here to convert to
    photon flux density per pixel before resampling and PSF convolution.

    ``incl_deg`` / ``pa_deg`` are the disk's intrinsic orientation in the
    sky frame; ``telescope_pa_deg`` is the telescope's roll. The disk is
    rendered at its intrinsic geometry, then resample_flux rotates the
    rendered image by ``-telescope_pa_deg`` into the detector frame.

    Raises:
        ValueError: if ``optical_path.coronagraph.psf_datacube`` is
            ``None``.
    """
    if optical_path.coronagraph.psf_datacube is None:
        raise ValueError(
            "disk_rate requires a coronagraph with a PSF "
            "datacube; got optical_path.coronagraph.psf_datacube=None. "
            "The disk pipeline convolves the resampled disk image with "
            "the per-source-position PSFs and cannot run without it."
        )

    contrast = disk.surface_brightness(wavelength_nm, start_time_jd, incl_deg, pa_deg)
    star_flux = star.spec_flux_density(wavelength_nm, start_time_jd)
    flux = contrast * star_flux
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    pixscale_tgt = lambda_d_to_arcsec(
        optical_path.coronagraph.pixel_scale_lod,
        wavelength_nm,
        optical_path.primary.diameter_m,
    )
    ny, nx = optical_path.coronagraph.psf_shape
    flux = resample_flux(
        flux,
        disk.pixel_scale_arcsec,
        pixscale_tgt,
        (ny, nx),
        -telescope_pa_deg,
    )

    psf_datacube = optical_path.coronagraph.psf_datacube
    n_src_y, n_src_x = psf_datacube.shape[:2]
    q_src_y = ny // 2 + 1
    q_src_x = nx // 2 + 1
    if n_src_y == ny and n_src_x == nx:
        image_rate_coro = jnp.einsum("ij,ijxy->xy", flux, psf_datacube)
    elif n_src_y == q_src_y and n_src_x == q_src_x:
        image_rate_coro = _convolve_quadrants(flux, psf_datacube)
    else:
        raise ValueError(
            "disk_rate: psf_datacube source-grid shape "
            f"({n_src_y}, {n_src_x}) does not match either the full PSF "
            f"shape ({ny}, {nx}) or the quarter PSF shape "
            f"({q_src_y}, {q_src_x}). Coronagraphs must publish a full "
            "or quarter datacube."
        )

    return post_coro_bin_processing(image_rate_coro, wavelength_nm, optical_path)


def disk_readout(
    disk,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    incl_deg,
    pa_deg,
):
    """Process a disk through the provided optical path.

    ``incl_deg`` / ``pa_deg`` are the disk's intrinsic sky-frame
    orientation; ``system_readout`` pulls them from
    ``scene.system.midplane_inc_deg`` / ``midplane_pa_deg`` so every
    disk component in the System renders at the same midplane.
    """
    image_rate_detector = disk_rate(
        disk,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        telescope_pa_deg=telescope_pa_deg,
        star=star,
        incl_deg=incl_deg,
        pa_deg=pa_deg,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Zodi
# ---------------------------------------------------------------------------


def zodi_rate(
    zodi: ZodiSource,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Generate the zodi count rate on the detector.

    Treats zodi as a spatially uniform surface-brightness source. The
    coronagraph's sky transmission map sets the per-pixel attenuation;
    no PSF convolution is needed (a flat field convolved with any
    normalised PSF returns itself).
    """
    sb_per_arcsec2 = zodi.spec_flux_density(
        wavelength_nm, start_time_jd, ecliptic_lat_deg, solar_lon_deg
    )
    pix_arcsec = lambda_d_to_arcsec(
        optical_path.coronagraph.pixel_scale_lod,
        wavelength_nm,
        optical_path.primary.diameter_m,
    )
    flux_per_pixel = sb_per_arcsec2 * pix_arcsec**2

    flux_map = flux_per_pixel * optical_path.coronagraph.sky_trans
    flux_map = pre_coro_bin_processing(
        flux_map, wavelength_nm, bin_width_nm, optical_path
    )
    return post_coro_bin_processing(flux_map, wavelength_nm, optical_path)


def zodi_readout(
    zodi: ZodiSource,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Process a zodi source through the provided optical path."""
    image_rate_detector = zodi_rate(
        zodi,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        ecliptic_lat_deg=ecliptic_lat_deg,
        solar_lon_deg=solar_lon_deg,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Whole-scene orchestrator
# ---------------------------------------------------------------------------


def system_rate(
    scene,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Sum of deterministic per-source count rates for a :class:`~skyscapes.Scene`.

    The differentiable companion to :func:`system_readout`. Returns the
    total rate map (electrons/s/pixel, no Poisson noise, no QE multiply)
    summing star, every planet, the optional disk, and the optional zodi.
    Use this for likelihood evaluation, retrievals, or any inference loop
    that needs gradients through the full forward model.
    """
    has_disk = scene.system.disk is not None
    has_zodi = scene.zodi is not None

    total = star_rate(
        scene.system.star,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
    )

    for planet in scene.system.planets:
        total = total + planet_rate(
            planet,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            trig_solver=scene.system.trig_solver,
        )

    if has_disk:
        total = total + disk_rate(
            scene.system.disk,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            incl_deg=jnp.asarray(scene.system.midplane_inc_deg),
            pa_deg=jnp.asarray(scene.system.midplane_pa_deg),
        )

    if has_zodi:
        total = total + zodi_rate(
            scene.zodi,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            ecliptic_lat_deg=ecliptic_lat_deg,
            solar_lon_deg=solar_lon_deg,
        )

    return total


def system_readout(
    scene,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Simulate a full :class:`~skyscapes.Scene` through the optical path.

    Sums per-source detector readouts. Each source consumes its own
    independent PRNG subkey (see :mod:`jax.random` best practices).

    The Python loop over ``scene.system.planets`` is intentionally
    unjitted -- it orchestrates JIT-cached per-Planet-type kernels (see
    ``brain/Planet Loop Architecture.md``). The expensive math is inside
    each ``planet_readout`` call, not the loop.
    """
    has_disk = scene.system.disk is not None
    has_zodi = scene.zodi is not None

    n_keys = 1 + len(scene.system.planets) + int(has_disk) + int(has_zodi)
    keys = iter(jax.random.split(prng_key, n_keys))

    total = star_readout(
        scene.system.star,
        optical_path,
        next(keys),
        start_time_jd=start_time_jd,
        exposure_time_s=exposure_time_s,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
    )

    for planet in scene.system.planets:
        total = total + planet_readout(
            planet,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            trig_solver=scene.system.trig_solver,
        )

    if has_disk:
        total = total + disk_readout(
            scene.system.disk,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            incl_deg=jnp.asarray(scene.system.midplane_inc_deg),
            pa_deg=jnp.asarray(scene.system.midplane_pa_deg),
        )

    if has_zodi:
        total = total + zodi_readout(
            scene.zodi,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            ecliptic_lat_deg=ecliptic_lat_deg,
            solar_lon_deg=solar_lon_deg,
        )

    return total
