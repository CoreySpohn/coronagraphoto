"""Functions for running full simulations and processing sources."""

import jax
import jax.numpy as jnp
from hwoutils.conversions import arcsec_to_lambda_d, lambda_d_to_arcsec
from hwoutils.transforms import ccw_rotation_matrix, resample_flux
from skyscapes.background import ZodiSource


def pre_coro_bin_processing(flux, bin_center_nm, bin_width_nm, optical_path):
    """Process a bin through the pre-coro elements of the optical path."""
    # Multiply by the wavelength bin width
    # ph/s/m^2
    flux = flux * bin_width_nm

    # Multiply by the primary aperture area
    # ph/s
    flux = optical_path.primary.apply(flux, bin_center_nm)

    # Multiply by the combined attenuation of the optical path
    # ph/s
    path_attenuation = optical_path.calculate_combined_attenuation(bin_center_nm)
    flux = flux * path_attenuation
    return flux


def post_coro_bin_processing(image_rate_coro, bin_center_nm, optical_path):
    """Process a bin through the post-coro elements of the optical path."""
    # Now map that to the detector pixels
    image_rate_detector = optical_path.detector.resample_to_detector(
        image_rate_coro,
        optical_path.coronagraph.pixel_scale_lod,
        bin_center_nm,
        optical_path.primary.diameter_m,
    )
    # Clip the image rate to be non-negative
    image_rate_detector = jnp.clip(image_rate_detector, 0, None)
    return image_rate_detector


def gen_planet_count_rate(
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    planet,
    optical_path,
    *,
    star,
    trig_solver,
):
    """Generate the per-batch planet count rate on the detector.

    Operates on a single ``skyscapes.scene.Planet`` (which internally
    batches K planets sharing the same atmosphere class). The Python
    loop over a heterogeneous ``System.planets`` tuple lives in
    :func:`sim_system`; this function stays inside the per-Planet-type
    JIT cache boundary (see ``brain/Planet Loop Architecture.md``).
    """
    # The new Planet API takes a 1-D time axis; squeeze T=1.
    source_positions_as = planet.position_arcsec(
        trig_solver, jnp.atleast_1d(start_time_jd), star=star
    )[:, :, 0]  # (2, K)

    # A positive telescope PA corresponds to a CW rotation of the sky.
    rotation_matrix = ccw_rotation_matrix(-telescope_pa_deg)
    source_positions_as = rotation_matrix @ source_positions_as

    # arcsec -> lambda/D.
    source_positions_lod = arcsec_to_lambda_d(
        source_positions_as, wavelength_nm, optical_path.primary.diameter_m
    )

    # Per-planet flux density. spec_flux_density returns (K, T); we pass T=1
    # (single epoch) and drop the T axis to get (K,).
    flux = planet.spec_flux_density(
        trig_solver,
        jnp.atleast_1d(wavelength_nm),
        jnp.atleast_1d(start_time_jd),
        star=star,
    )[:, 0]  # (K,) -- drop T=1 axis
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    # Create the PSFs for each source position.
    psfs = optical_path.coronagraph.create_psfs(
        source_positions_lod[0], source_positions_lod[1]
    )
    # Multiply flux by the coronagraph PSF and sum over planets.
    image_rate_coro = jnp.einsum("i,ijk->jk", flux, psfs)

    # Map to detector.
    return post_coro_bin_processing(image_rate_coro, wavelength_nm, optical_path)


def sim_planets(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    planet,
    optical_path,
    prng_key,
    *,
    star,
    trig_solver,
):
    """Process a per-batch Planet through the optical path."""
    image_rate_detector = gen_planet_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        telescope_pa_deg,
        planet,
        optical_path,
        star=star,
        trig_solver=trig_solver,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


def gen_star_count_rate(
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    star,
    optical_path,
):
    """Generate the star count rate on the detector."""
    # Convert the source diameter from arcseconds to lambda/D
    source_diam_lod = arcsec_to_lambda_d(
        star.diameter_arcsec, wavelength_nm, optical_path.primary.diameter_m
    )
    # Get the source's spectral flux density at the bin center wavelength
    # ph/s/m^2/nm
    flux = star.spec_flux_density(wavelength_nm, start_time_jd)

    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    # Multiply by the coronagraph PSF to get the image rate
    # ph/s/pixel
    image_rate_coro = optical_path.coronagraph.stellar_intens(source_diam_lod) * flux

    # Map to detector
    image_rate_detector = post_coro_bin_processing(
        image_rate_coro, wavelength_nm, optical_path
    )
    return image_rate_detector


def sim_star(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    star,
    optical_path,
    prng_key,
):
    """Process a star through the provided optical path."""
    image_rate_detector = gen_star_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        star,
        optical_path,
    )
    readout_electrons = optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )

    return readout_electrons


def _convolve_quadrants(flux, psf_datacube):
    """Convolve flux with a quarter-symmetric PSF datacube.

    Uses a fold-and-sum approach.

    Handles padding dynamically to ensure all quadrants match the shape of the
    first quadrant (which defines the PSF datacube shape).
    """
    ny, nx = flux.shape

    # Define center to match create_psf_datacube logic: (N-1)//2
    cy, cx = (ny - 1) // 2, (nx - 1) // 2

    # Q1: top-right (includes center pixel and axes) -> reference shape
    # Shape: (ny - cy, nx - cx)
    q1 = flux[cy:, cx:]
    target_h, target_w = q1.shape

    # Q2: top-left (excludes center column)
    # Original Shape: (ny - cy, cx)
    # Flip X, pad inner (left) with 1 zero (for axis),
    # and pad outer (right) to match target width.
    q2_raw = flux[cy:, :cx]
    q2_flipped = q2_raw[:, ::-1]

    pad_q2_right = target_w - (q2_flipped.shape[1] + 1)
    pad_q2_right = max(0, pad_q2_right)

    q2 = jnp.pad(q2_flipped, ((0, 0), (1, pad_q2_right)))

    # Q3: bottom-left (excludes center row and column)
    # Original Shape: (cy, cx)
    # flip X and Y, pad top (inner), left (inner), and bottom/right (outer).
    q3_raw = flux[:cy, :cx]
    q3_flipped = q3_raw[::-1, ::-1]

    pad_q3_bottom = target_h - (q3_flipped.shape[0] + 1)
    pad_q3_right = target_w - (q3_flipped.shape[1] + 1)
    pad_q3_bottom = max(0, pad_q3_bottom)
    pad_q3_right = max(0, pad_q3_right)

    q3 = jnp.pad(q3_flipped, ((1, pad_q3_bottom), (1, pad_q3_right)))

    # Q4: bottom-right (excludes center row)
    # Original Shape: (cy, nx - cx)
    # flip Y, pad top (inner) and bottom (outer).
    q4_raw = flux[:cy, cx:]
    q4_flipped = q4_raw[::-1, :]

    pad_q4_bottom = target_h - (q4_flipped.shape[0] + 1)
    pad_q4_bottom = max(0, pad_q4_bottom)

    q4 = jnp.pad(q4_flipped, ((1, pad_q4_bottom), (0, 0)))

    # Stack for batched processing
    # all q arrays should now be (target_h, target_w)
    flux_stack = jnp.stack([q1, q2, q3, q4])

    # Batched Einsum
    partial_images = jnp.einsum("qij,ijxy->qxy", flux_stack, psf_datacube)

    # Unflip and sum
    img_q1 = partial_images[0]
    img_q2 = jnp.fliplr(partial_images[1])
    img_q3 = jnp.flipud(jnp.fliplr(partial_images[2]))
    img_q4 = jnp.flipud(partial_images[3])

    return img_q1 + img_q2 + img_q3 + img_q4


def gen_disk_count_rate(
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    disk,
    optical_path,
    *,
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
            ``None``. The disk pipeline convolves the resampled disk
            image with the per-source-position PSFs and cannot run
            without it.
    """
    if optical_path.coronagraph.psf_datacube is None:
        raise ValueError(
            "gen_disk_count_rate requires a coronagraph with a PSF "
            "datacube; got optical_path.coronagraph.psf_datacube=None. "
            "The disk pipeline convolves the resampled disk image with "
            "the per-source-position PSFs and cannot run without it."
        )

    # ph/s/m^2/nm per pixel = contrast * star flux density.
    # SpectrumStar.spec_flux_density returns a scalar when given scalar
    # inputs (same call shape that gen_star_count_rate uses today).
    contrast = disk.surface_brightness(wavelength_nm, start_time_jd, incl_deg, pa_deg)
    star_flux = star.spec_flux_density(wavelength_nm, start_time_jd)
    flux = contrast * star_flux

    # ph/s/pixel after bandwidth + collecting area + optics attenuation.
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    # Resample disk pixel scale -> coronagraph pixel scale; rotate by telescope PA.
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

    # Existing convolution paths (full vs quarter PSF datacube).
    psf_datacube = optical_path.coronagraph.psf_datacube
    n_src_y, n_src_x = psf_datacube.shape[:2]
    q_src_y = ny // 2 + 1
    q_src_x = nx // 2 + 1
    if n_src_y == ny and n_src_x == nx:
        # Full PSF datacube
        image_rate_coro = jnp.einsum("ij,ijxy->xy", flux, psf_datacube)
    elif n_src_y == q_src_y and n_src_x == q_src_x:
        # Quarter PSF datacube
        image_rate_coro = _convolve_quadrants(flux, psf_datacube)
    else:
        raise ValueError(
            "gen_disk_count_rate: psf_datacube source-grid shape "
            f"({n_src_y}, {n_src_x}) does not match either the full PSF "
            f"shape ({ny}, {nx}) or the quarter PSF shape "
            f"({q_src_y}, {q_src_x}). Coronagraphs must publish a full "
            "or quarter datacube."
        )

    # Map to detector.
    return post_coro_bin_processing(image_rate_coro, wavelength_nm, optical_path)


def sim_disk(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    disk,
    optical_path,
    prng_key,
    *,
    star,
    incl_deg,
    pa_deg,
):
    """Process a disk through the provided optical path.

    ``incl_deg`` / ``pa_deg`` are the disk's intrinsic sky-frame
    orientation; ``sim_system`` pulls them from
    ``scene.system.midplane_inc_deg`` / ``midplane_pa_deg`` so every
    disk component in the System renders at the same midplane.
    """
    image_rate_detector = gen_disk_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        telescope_pa_deg,
        disk,
        optical_path,
        star=star,
        incl_deg=incl_deg,
        pa_deg=pa_deg,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


def gen_zodi_count_rate(
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    zodi: ZodiSource,
    optical_path,
    ecliptic_lat_deg: float = 0.0,
    solar_lon_deg: float = 135.0,
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
    pix_solid_angle_arcsec2 = pix_arcsec**2

    flux_per_pixel = sb_per_arcsec2 * pix_solid_angle_arcsec2

    sky_trans = optical_path.coronagraph.sky_trans()
    flux_map = flux_per_pixel * sky_trans

    flux_map = pre_coro_bin_processing(
        flux_map, wavelength_nm, bin_width_nm, optical_path
    )

    image_rate_detector = post_coro_bin_processing(
        flux_map, wavelength_nm, optical_path
    )
    return image_rate_detector


def sim_zodi(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    zodi: ZodiSource,
    optical_path,
    prng_key,
    ecliptic_lat_deg: float = 0.0,
    solar_lon_deg: float = 135.0,
):
    """Process a zodi source through the provided optical path."""
    image_rate_detector = gen_zodi_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        zodi,
        optical_path,
        ecliptic_lat_deg=ecliptic_lat_deg,
        solar_lon_deg=solar_lon_deg,
    )
    readout_electrons = optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )
    return readout_electrons


def sim_system(
    scene,
    optical_path,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    prng_key,
):
    """Simulate a full :class:`~skyscapes.Scene` through the optical path.

    Sums per-source detector readouts. Each source consumes its own
    independent PRNG subkey (see :mod:`jax.random` best practices).

    The Python loop over ``scene.system.planets`` is intentionally
    unjitted -- it orchestrates JIT-cached per-Planet-type kernels (see
    ``brain/Planet Loop Architecture.md``). The expensive math is inside
    each ``sim_planets`` call, not the loop.
    """
    has_disk = scene.system.disk is not None
    has_zodi = scene.zodi is not None

    n_keys = 1 + len(scene.system.planets) + int(has_disk) + int(has_zodi)
    keys = jax.random.split(prng_key, n_keys)
    idx = 0

    total = sim_star(
        start_time_jd,
        exposure_time_s,
        wavelength_nm,
        bin_width_nm,
        scene.system.star,
        optical_path,
        keys[idx],
    )
    idx += 1

    for planet in scene.system.planets:
        total = total + sim_planets(
            start_time_jd,
            exposure_time_s,
            wavelength_nm,
            bin_width_nm,
            telescope_pa_deg,
            planet,
            optical_path,
            keys[idx],
            star=scene.system.star,
            trig_solver=scene.system.trig_solver,
        )
        idx += 1

    if has_disk:
        total = total + sim_disk(
            start_time_jd,
            exposure_time_s,
            wavelength_nm,
            bin_width_nm,
            telescope_pa_deg,
            scene.system.disk,
            optical_path,
            keys[idx],
            star=scene.system.star,
            incl_deg=jnp.asarray(scene.system.midplane_inc_deg),
            pa_deg=jnp.asarray(scene.system.midplane_pa_deg),
        )
        idx += 1

    if has_zodi:
        total = total + sim_zodi(
            start_time_jd,
            exposure_time_s,
            wavelength_nm,
            bin_width_nm,
            scene.zodi,
            optical_path,
            keys[idx],
        )
        idx += 1

    return total
