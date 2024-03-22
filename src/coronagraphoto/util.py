from itertools import combinations
from pathlib import PurePath

import astropy.units as u
import numpy as np
from astropy.time import Time
from lod_unit import lod_eq
from scipy.ndimage import zoom


def gen_wavelength_grid(bandpass, resolution):
    """
    Generates wavelengths that sample a Synphot bandpass at a given resolution.

    This function calculates wavelengths within the specified bandpass range
    such that each wavelength interval corresponds to a constant spectral resolution.
    The spectral resolution is defined as R = λ/Δλ, where λ is the wavelength and Δλ
    is the wavelength interval. The function iteratively adds these intervals starting
    from the lower limit of the bandpass until it reaches or surpasses the upper limit.

    Args:
        bandpass (synphot.SpectralElement):
            A synphot bandpass object with the "waveset" attribute that
            indicates the wavelengths necessary to sample the bandpass.
        resolution (float):
            The desired constant spectral resolution (R).
    Returns:
        wavelengths (astropy.units.quantity.Quantity):
            An array of wavelengths sampled across the bandpass at the
            specified resolution.
        delta_lambdas (astropy.units.quantity.Quantity):
            An array of wavelength intervals that correspond to the
            specified resolution.
    """
    first_wavelength = bandpass.waveset[0]
    last_wavelength = bandpass.waveset[-1]
    wavelengths = []
    delta_lambdas = []
    current_wavelength = first_wavelength
    while current_wavelength < last_wavelength:
        wavelengths.append(current_wavelength.value)
        delta_lambda = current_wavelength / resolution
        current_wavelength += delta_lambda
        delta_lambdas.append(delta_lambda.value)
    wavelengths = (np.array(wavelengths) * first_wavelength.unit).to(u.nm)
    delta_lambdas = (np.array(delta_lambdas) * first_wavelength.unit).to(u.nm)
    return wavelengths, delta_lambdas


def convert_pixels(unit, obs, plane="coro"):
    if plane == "coro":
        pix_arr = np.arange(obs.coronagraph.npixels)
        lam = obs.scenario.central_wavelength
        diameter = obs.scenario.diameter
        pix_scale = (obs.coronagraph.pixel_scale * u.pix).to(
            u.rad, lod_eq(lam, diameter)
        ) / u.pix
        pix_shape = [obs.coronagraph.npixels] * 2
    elif plane == "det":
        pix_arr = np.arange(obs.scenario.detector_shape[0])
        pix_scale = obs.scenario.detector_pixel_scale
        pix_shape = obs.scenario.detector_shape

    xnpix, ynpix = pix_shape
    star_pixel = (xnpix / 2, ynpix / 2)
    if unit.physical_type == "angle":
        pix_scale = (pix_scale).to(unit / u.pix)
        xunit_arr = (pix_arr - star_pixel[0]) * u.pix * pix_scale
        yunit_arr = (pix_arr - star_pixel[1]) * u.pix * pix_scale
    else:
        raise NotImplementedError("Conversion to this unit not implemented.")
    return xunit_arr, yunit_arr


def resample_single_image(image, lod_scale, wavelength, diam, det_shape, det_scale):
    """
    Resample a single image from lambda/D units to arcseconds based on the
    detector's shape and pixel scale.

    Args:
        image (numpy.ndarray):
            The single image to be resampled.
        lod_scale (astropy.units.quantity.Quantity):
            The scale of the image given. Must be lod/u.pix.
        wavelength (astropy.units.quantity.Quantity):
            The wavelength of the image.
        diam (astropy.units.quantity.Quantity):
            The diameter of the telescope.
        det_shape (tuple):
            The shape of the detector in pixels.
        det_scale (astropy.units.quantity.Quantity):
            The pixel scale of the detector. Must be u.arcsec/u.pix.

    Returns:
        numpy.ndarray: The resampled single image.
    """
    # Convert the scale from lod per pixel to arcseconds per pixel
    lod_scale_in_arcsec = (lod_scale * u.pix).to(
        u.arcsec, lod_eq(wavelength, diam)
    ) / u.pix

    # Calculate the zoom factor for resampling
    zoom_factor = (lod_scale_in_arcsec / det_scale).decompose().value

    # Resample (zoom) the image based on the calculated zoom factor
    scaled_image = zoom(image, zoom_factor, mode="nearest", order=5)

    # Calculate how much the resampled image needs to be cropped or padded
    # to match the desired detector shape
    center_offset = (np.array(scaled_image.shape) - np.array(det_shape)) / 2

    # Check if the resampled image needs padding
    # (if it's smaller than the desired shape)
    if np.any(center_offset < 0):
        pad_amount = np.abs(np.minimum(center_offset, 0)).astype(int)

        # Pad the image symmetrically on both sides
        padded_image = np.pad(
            scaled_image,
            ((pad_amount[0], pad_amount[0]), (pad_amount[1], pad_amount[1])),
            mode="constant",
        )

        # Crop or further pad the image as necessary to match detector shape
        final_image = (
            padded_image[: det_shape[0], : det_shape[1]]
            if padded_image.shape[0] > det_shape[0]
            else np.pad(
                padded_image,
                (
                    (0, det_shape[0] - padded_image.shape[0]),
                    (0, det_shape[1] - padded_image.shape[1]),
                ),
                mode="constant",
            )
        )
    else:
        # If no padding is needed, just crop the image to the desired shape
        final_image = scaled_image[
            int(center_offset[0]) : int(center_offset[0] + det_shape[0]),
            int(center_offset[1]) : int(center_offset[1] + det_shape[1]),
        ]

    return final_image


def get_detector_images(lod_arr, lod_scale, lam, D, det_shape, det_scale):
    """
    Resample multiple frames in lambda/D units to arcseconds based on the
    detector's shape and pixel scale. Handles both single and multiple
    wavelength frames.

    Args:
        lod_arr (numpy.ndarray):
            The array of images to be resampled. Shape can be either
            (nframes, nxpix, nypix) or (nframes, nlambda, nxpix, nypix).
        lod_scale (astropy.units.quantity.Quantity):
            The scale of the images given. Must be lod/u.pix.
        lam (astropy.units.quantity.Quantity):
            The wavelength(s) of the images. Can be a scalar or an array with
            length nlambda.
        D (astropy.units.quantity.Quantity):
            The diameter of the telescope.
        det_shape (tuple):
            The shape of the detector in pixels (height, width).
        det_scale (astropy.units.quantity.Quantity):
            The pixel scale of the detector. Must be u.arcsec/u.pix.

    Returns:
        final_image (numpy.ndarray):
            The array of resampled images. Shape will be
            (nframes, det_shape[0], det_shape[1]) for single wavelength per frame or
            (nframes, nlambda, det_shape[0], det_shape[1]) for multiple wavelengths
            per frame.
    """
    nframes = lod_arr.shape[0]
    has_wavelength_dim = len(lod_arr.shape) == 4

    if has_wavelength_dim:
        nlambda = lod_arr.shape[1]
        final_images = np.zeros((nframes, nlambda, det_shape[0], det_shape[1]))

        # Validate the length of lam if it's not a scalar
        if not lam.isscalar and len(lam) != nlambda:
            raise ValueError("Length of lam must match nlambda in lod_arr")

        for frame_idx in range(nframes):
            for lambda_idx in range(nlambda):
                wavelength = lam if lam.isscalar else lam[lambda_idx]
                final_images[frame_idx, lambda_idx] = resample_single_image(
                    lod_arr[frame_idx, lambda_idx],
                    lod_scale,
                    wavelength,
                    D,
                    det_shape,
                    det_scale,
                )
    else:
        final_images = np.zeros((nframes, det_shape[0], det_shape[1]))

        # Use the scalar lam for all frames
        for frame_idx in range(nframes):
            final_images[frame_idx] = resample_single_image(
                lod_arr[frame_idx], lod_scale, lam, D, det_shape, det_scale
            )

    return final_images


def find_distinguishing_attributes(*observations):
    """
    Finds and returns the attributes that distinguish each given Observation
    instance from the others.

    This function compares each Observation instance against all others and
    identifies the unique attributes that set each instance apart. An attribute
    is considered unique for an instance if it differs from the same attribute
    in all other instances.

    Args:
        *observations:
            An arbitrary number of Observation instances.

    Returns:
        distinguishing_attrs (dict):
            A dictionary where each key is an Observation instance and the
            value is a dictionary of attributes that uniquely identify this
            instance among the provided instances.
    """
    # Whitelist of attributes to ignore
    whitelist = [
        "system",
        "coronagraph",
        "observing_scenario",
        "bandpass",
        "star_count_rate",
        "planet_count_rate",
        "disk_count_rate",
        "illuminated_area",
        "transmission",
        "bandwidth",
        "bandpass_model",
        "psf_datacube",
        "full_bandwidth",
        "spectral_wavelength",
        "frame_start_times",
        "spectral_wavelength_grid",
        "spectral_bandwidths",
        "spectral_transmission",
    ]
    whitelisted_types = []

    # Dictionary to hold distinguishing attributes for each observation
    distinguishing_attrs = {obs: {} for obs in observations}
    attr_names = set()
    attr_values = {}

    # Identify attributes that differ between any two observations
    differing_attrs = set()
    for obs1, obs2 in combinations(observations, 2):
        for attr in obs1.__dict__:
            if attr in whitelist or type(getattr(obs1, attr)) in whitelisted_types:
                continue
            elif isinstance(getattr(obs1, attr), PurePath):
                continue
            if getattr(obs1, attr) != getattr(obs2, attr):
                differing_attrs.add(attr)

    # Evaluate combinations of differing attributes for uniqueness
    for r in range(1, len(differing_attrs) + 1):
        for attr_combination in combinations(differing_attrs, r):
            for obs in observations:
                if is_unique_combination(obs, attr_combination, observations):
                    for attr in attr_combination:
                        distinguishing_attrs[obs][attr] = getattr(obs, attr)
                        attr_names.add(attr)
                        if attr not in attr_values:
                            attr_values[attr] = set()
                        attr_values[attr].add(getattr(obs, attr))

    for attr, values in attr_values.items():
        sorted_values = sorted(values)
        if type(sorted_values[0]) == u.Quantity:
            replacement = u.Quantity(sorted_values)
        elif type(sorted_values[0]) == Time:
            replacement = Time(sorted_values).datetime64
        elif isinstance(sorted_values[0], str):
            replacement = sorted_values
        else:
            raise NotImplementedError("Add support for this type")
        attr_values[attr] = replacement

    return distinguishing_attrs, list(attr_names), attr_values


def is_unique_combination(obs, attr_combination, all_observations):
    """
    Check if a combination of attributes is unique for an observation.
    """
    for other_obs in all_observations:
        if other_obs != obs and all(
            getattr(other_obs, attr, None) == getattr(obs, attr, None)
            for attr in attr_combination
        ):
            return False
    return True
