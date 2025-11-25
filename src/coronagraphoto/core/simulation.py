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

    # Multiply by the PSF datacube to get the image rate of the disk
    # ph/s/pixel
    image_rate_coro = jnp.einsum(
        "ij,ijxy->xy", flux, optical_path.coronagraph.psf_datacube
    )

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
