"""Utility functions for coronagraph observation simulations.

This module provides utility functions for coronagraph observation simulations,
including wavelength grid generation, pixel coordinate conversion, image
resampling, and attribute comparison utilities.

The module contains functions for:
- Spectral grid generation with constant resolution
- Pixel coordinate conversion between different planes (coronagraph/detector)
- Image resampling with flux conservation
- Multi-dimensional image processing for detector conversion
- Observation attribute comparison and uniqueness detection

Key Features:
- Wavelength grid generation with spectral resolution control
- Coordinate system conversion between lambda/D and detector pixels
- Flux-conserving image resampling and zooming
- Efficient multi-frame image processing
- Robust attribute comparison for observation datasets

The module uses Astropy for unit handling, NumPy for array operations,
and SciPy for image processing functions.
"""

from itertools import combinations
from pathlib import PurePath

import astropy.units as u
import numpy as np
from astropy.time import Time
from lod_unit import lod_eq
from scipy.ndimage import shift, zoom


def gen_wavelength_grid(bandpass, resolution):
    """Generate wavelengths that sample a Synphot bandpass at a given resolution.

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
    """Convert pixel coordinates to physical units for coronagraph or detector planes.

    This function converts pixel coordinates to physical units (typically angular
    units) for either the coronagraph plane (lambda/D units) or detector plane
    (arcseconds). The conversion accounts for the pixel scale and centers the
    coordinate system on the star position.

    Args:
        unit (astropy.units.Unit):
            The target unit for the coordinate conversion. Must have physical
            type "angle" (e.g., u.arcsec, u.mas, u.rad).
        obs (Observation):
            Observation object containing coronagraph and scenario information.
        plane (str, optional):
            The plane to convert coordinates for. Options are:
            - "coro": Coronagraph plane (lambda/D units)
            - "det": Detector plane (arcseconds)
            Default is "coro".

    Returns:
        xunit_arr (astropy.units.quantity.Quantity):
            Array of x-coordinates in the specified unit, centered on the star.
        yunit_arr (astropy.units.quantity.Quantity):
            Array of y-coordinates in the specified unit, centered on the star.

    Raises:
        NotImplementedError:
            If the target unit does not have physical type "angle".

    Note:
        The coordinate system is centered on the star position, which is
        assumed to be at the center of the pixel array.
    """
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
    """Resample a single image from lambda/D units to detector pixel scale.

    This function resamples a single image from coronagraph pixel scale (lambda/D)
    to detector pixel scale (arcseconds) using bicubic interpolation. The function
    handles both up-sampling and down-sampling cases, with proper cropping or
    padding to match the target detector shape.

    The resampling process:
    1. Converts lambda/D scale to arcseconds using the wavelength and telescope diameter
    2. Calculates the zoom factor needed for the conversion
    3. Applies bicubic interpolation to resample the image
    4. Crops or pads the result to match the detector shape

    Args:
        image (numpy.ndarray):
            The single image to be resampled in coronagraph pixels.
        lod_scale (astropy.units.quantity.Quantity):
            The pixel scale of the input image in lambda/D per pixel.
        wavelength (astropy.units.quantity.Quantity):
            The wavelength of the observation.
        diam (astropy.units.quantity.Quantity):
            The telescope diameter.
        det_shape (tuple):
            The target detector shape in pixels (height, width).
        det_scale (astropy.units.quantity.Quantity):
            The target detector pixel scale in arcseconds per pixel.

    Returns:
        numpy.ndarray:
            The resampled image with shape matching det_shape.

    Note:
        The function uses bicubic interpolation (order=5) for high-quality
        resampling. For very small zoom factors (down-sampling), interpolation
        artifacts may occur.
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
    """Resample multiple frames from lambda/D units to detector pixel scale.

    This function handles resampling of multiple image frames from coronagraph
    pixel scale (lambda/D) to detector pixel scale (arcseconds). It supports
    both single-wavelength and multi-wavelength observations, automatically
    detecting the input array shape and processing accordingly.

    The function processes each frame individually using resample_single_image,
    handling both 3D arrays (nframes, nxpix, nypix) for single-wavelength
    observations and 4D arrays (nframes, nlambda, nxpix, nypix) for
    multi-wavelength observations.

    Args:
        lod_arr (numpy.ndarray):
            The array of images to be resampled. Shape can be either:
            - (nframes, nxpix, nypix) for single wavelength per frame
            - (nframes, nlambda, nxpix, nypix) for multiple wavelengths per frame
        lod_scale (astropy.units.quantity.Quantity):
            The pixel scale of the input images in lambda/D per pixel.
        lam (astropy.units.quantity.Quantity):
            The wavelength(s) of the images. Can be a scalar for single-wavelength
            observations or an array with length nlambda for multi-wavelength.
        D (astropy.units.quantity.Quantity):
            The telescope diameter.
        det_shape (tuple):
            The target detector shape in pixels (height, width).
        det_scale (astropy.units.quantity.Quantity):
            The target detector pixel scale in arcseconds per pixel.

    Returns:
        numpy.ndarray:
            The array of resampled images. Shape will be:
            - (nframes, det_shape[0], det_shape[1]) for single wavelength
            - (nframes, nlambda, det_shape[0], det_shape[1]) for multiple wavelengths

    Raises:
        ValueError:
            If the length of lam array doesn't match nlambda in lod_arr for
            multi-wavelength observations.

    Note:
        For multi-wavelength observations, each wavelength is processed with
        its corresponding wavelength value from the lam array.
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
    """Find attributes that distinguish each Observation instance from others.

    This function compares multiple Observation instances and identifies the
    unique attributes that set each instance apart from the others. It uses
    a whitelist approach to ignore common attributes that don't distinguish
    between observations and evaluates combinations of attributes to find
    the minimal set needed for uniqueness.

    The function performs the following steps:
    1. Compares each observation against all others to find differing attributes
    2. Evaluates combinations of differing attributes to find unique combinations
    3. Returns the distinguishing attributes for each observation along with
       the complete set of attribute names and their possible values

    Args:
        *observations:
            An arbitrary number of Observation instances to compare.

    Returns:
        tuple:
            A tuple containing three elements:
            - distinguishing_attrs (dict):
                Dictionary mapping each Observation to its distinguishing attributes
            - attr_names (list):
                List of all attribute names that distinguish observations
            - attr_values (dict):
                Dictionary mapping attribute names to their possible values across
                all observations

    Note:
        The function uses a whitelist to ignore common attributes like system,
        coronagraph, and computed properties that don't distinguish between
        observations. It also handles special types like astropy.units.Quantity
        and astropy.time.Time appropriately.
    """
    # Whitelist of attributes to ignore
    whitelist = [
        "system",
        "coronagraph",
        "observing_scenario",
        "scenario",
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
    """Check if a combination of attributes is unique for an observation.

    This helper function determines whether a specific combination of attributes
    uniquely identifies an observation among all provided observations. It
    compares the attribute values of the target observation against all other
    observations to ensure no other observation has the same combination of
    attribute values.

    Args:
        obs (Observation):
            The observation to check for uniqueness.
        attr_combination (tuple):
            Tuple of attribute names to check for uniqueness.
        all_observations (list):
            List of all observations to compare against.

    Returns:
        bool:
            True if the combination of attributes is unique for the observation,
            False otherwise.

    Note:
        The function uses getattr with a default value of None to handle cases
        where an observation might not have a particular attribute.
    """
    for other_obs in all_observations:
        if other_obs != obs and all(
            getattr(other_obs, attr, None) == getattr(obs, attr, None)
            for attr in attr_combination
        ):
            return False
    return True


def zoom_conserve_flux(image, zoom_factor):
    """Image zoom that approximately preserves the total flux.

    This function performs bicubic resampling of an image while attempting to
    preserve the total flux. It uses scipy's zoom function with bicubic
    interpolation and then applies a normalization factor to compensate for
    the flux changes introduced by the interpolation.

    The normalization factor is calculated as 1/zoom_factor² to account for
    the area scaling introduced by the zoom operation. However, due to
    interpolation errors, the total flux is not exactly preserved, especially
    for very small zoom factors (down-sampling).

    Args:
        image (numpy.ndarray):
            Real-valued array containing the image information.
        zoom_factor (float):
            Linear zoom factor. Values < 1 down-sample the image, values > 1
            up-sample the image.

    Returns:
        numpy.ndarray:
            Resampled image with approximately preserved total flux.

    Note:
        The function uses bicubic interpolation (order=3) with zero padding
        outside the data boundaries. For very small zoom factors, interpolation
        artifacts may significantly affect flux conservation.
    """
    # Bicubic resample with zero padding outside the data
    out = zoom(image, zoom_factor, order=3, mode="constant", cval=0.0, prefilter=True)

    # Renormalization factor to preserve the total flux given how the zoom
    # function works
    norm_factor = 1 / zoom_factor**2

    return out * norm_factor
