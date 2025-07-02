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

as_per_pix = u.arcsec / u.pix


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
