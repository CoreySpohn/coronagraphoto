"""Functions for running full simulations and processing sources."""

import jax.numpy as jnp

import coronagraphoto.conversions as conv
from coronagraphoto.transforms.image_transforms import (
    ccw_rotation_matrix,
    resample_flux,
)


def pre_coro_bin_processing(flux, bin_center_nm, bin_width_nm, optical_path):
    """Process a bin through the pre-coro elements of the optical path."""
    # Multiply by the wavelength bin width
    # ph/s/m^2
    flux *= bin_width_nm

    # Multiply by the primary aperture area
    # ph/s
    flux = optical_path.primary.apply(flux, bin_center_nm)

    # Multiply by the combined attenuation of the optical path
    # ph/s
    path_attenuation = optical_path.calculate_combined_attenuation(bin_center_nm)
    flux *= path_attenuation
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
    position_angle_deg,
    planets,
    optical_path,
):
    """Generate the planet count rate on the detector."""
    # The on-sky position is calculated in RA vs Dec, and strictly in arcsec
    source_positions_as = planets.position(start_time_jd)

    # A positive position angle corresponds to a CW rotation of the sky
    rotation_matrix = ccw_rotation_matrix(-position_angle_deg)
    source_positions_as = rotation_matrix @ source_positions_as

    # Convert the source position from arcseconds to lambda/D
    source_positions_lod = conv.arcsec_to_lambda_d(
        source_positions_as, wavelength_nm, optical_path.primary.diameter_m
    )

    # Get the source's spectral flux density at the bin center wavelength
    # ph/s/m^2/nm
    flux = planets.spec_flux_density(wavelength_nm, start_time_jd)
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    # Create the PSFs for each source position
    psfs = optical_path.coronagraph.create_psfs(
        source_positions_lod[0], source_positions_lod[1]
    )
    # Multiply flux by the coronagraph PSF and sum to get the image rate
    # of all planets at once
    # ph/s/pixel
    image_rate_coro = jnp.einsum("i,ijk->jk", flux, psfs)

    # Map to detector
    image_rate_detector = post_coro_bin_processing(
        image_rate_coro, wavelength_nm, optical_path
    )

    return image_rate_detector


def sim_planets(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    position_angle_deg,
    planets,
    optical_path,
    prng_key,
):
    """Process an off-axis source through the provided optical path."""
    image_rate_detector = gen_planet_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        position_angle_deg,
        planets,
        optical_path,
    )
    # Readout the electrons from the detector
    readout_electrons = optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )
    return readout_electrons


def gen_star_count_rate(
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    star,
    optical_path,
):
    """Generate the star count rate on the detector."""
    # Convert the source diameter from arcseconds to lambda/D
    source_diam_lod = conv.arcsec_to_lambda_d(
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
    """Convolve flux with a quarter-symmetric PSF datacube using a fold-and-sum approach.

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
    position_angle_deg,
    disk,
    optical_path,
):
    """Generate the disk count rate on the detector."""
    # Get the source's spectral flux density at the bin center wavelength
    # ph/s/m^2/nm
    flux = disk.spec_flux_density(wavelength_nm, start_time_jd)

    # This gets us to ph/s/pixel for the disk
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    # Resample the disk image to the coronagraph pixel scale
    # Get source pixel scale in arcsec
    pixscale_src = disk.pixel_scale_arcsec
    # Calculate target pixel scale in arcsec (coronagraph pixel scale is in lambda/D)
    pixscale_tgt = conv.lambda_d_to_arcsec(
        optical_path.coronagraph.pixel_scale_lod,
        wavelength_nm,
        optical_path.primary.diameter_m,
    )
    # Get target shape from coronagraph PSF shape
    ny, nx = optical_path.coronagraph.psf_shape
    shape_tgt = (ny, nx)

    # Resample the flux while conserving total flux and rotating the disk
    flux = resample_flux(
        flux,
        pixscale_src,
        pixscale_tgt,
        shape_tgt,
        -position_angle_deg,
    )

    # Select convolution method based on PSF datacube shape
    psf_datacube = optical_path.coronagraph.psf_datacube

    # Dimensions of the PSF source grid
    n_src_y, n_src_x = psf_datacube.shape[:2]

    # Expected dimensions for quarter-grid optimization
    # (Assuming odd dimensions for full grid, quarter grid is (N-1)/2 + 1)
    # e.g., Full=101 -> Quarter=51
    q_src_y = ny // 2 + 1
    q_src_x = nx // 2 + 1

    if n_src_y == ny and n_src_x == nx:
        # Full PSF datacube
        # ph/s/pixel
        image_rate_coro = jnp.einsum("ij,ijxy->xy", flux, psf_datacube)
    elif n_src_y == q_src_y and n_src_x == q_src_x:
        # Quarter PSF datacube
        image_rate_coro = _convolve_quadrants(flux, psf_datacube)

    # Map to detector
    image_rate_detector = post_coro_bin_processing(
        image_rate_coro, wavelength_nm, optical_path
    )
    return image_rate_detector


def sim_disk(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    position_angle_deg,
    disk,
    optical_path,
    prng_key,
):
    """Process a disk through the provided optical path."""
    image_rate_detector = gen_disk_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        position_angle_deg,
        disk,
        optical_path,
    )
    readout_electrons = optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )
    return readout_electrons


def gen_zodi_count_rate(
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    zodi,
    optical_path,
):
    """Generate the zodiacal light count rate on the detector.

    Uses the coronagraph's sky transmission map (sky_trans) to modulate the
    uniform zodiacal light surface brightness.
    """
    # Get the source's spectral flux density at the bin center wavelength
    # ph/s/m^2/nm/arcsec^2
    flux = zodi.spec_flux_density(wavelength_nm, start_time_jd)

    # ph/s/arcsec^2
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    # Get coronagraph sky transmission
    sky_trans = optical_path.coronagraph.sky_trans

    # Calculate coronagraph pixel scale in arcseconds
    pixscale_coro_arcsec = conv.lambda_d_to_arcsec(
        optical_path.coronagraph.pixel_scale_lod,
        wavelength_nm,
        optical_path.primary.diameter_m,
    )

    # Calculate pixel area in arcsec^2
    pixel_area_arcsec2 = pixscale_coro_arcsec**2

    # Calculate image rate on coronagraph grid
    # ph/s/pixel = (ph/s/arcsec^2) * (unitless throughput) * (arcsec^2/pixel)
    image_rate_coro = flux * sky_trans * pixel_area_arcsec2

    # Map to detector
    image_rate_detector = post_coro_bin_processing(
        image_rate_coro, wavelength_nm, optical_path
    )

    return image_rate_detector


def sim_zodi(
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    zodi,
    optical_path,
    prng_key,
):
    """Process zodiacal light through the provided optical path."""
    image_rate_detector = gen_zodi_count_rate(
        start_time_jd,
        wavelength_nm,
        bin_width_nm,
        zodi,
        optical_path,
    )
    readout_electrons = optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )
    return readout_electrons
