"""Batch observation simulation for multiple times and wavelengths.

This module provides the Observations class for efficiently generating multiple
coronagraph observations across different times and wavelengths. It handles the
coordination of multiple Observation objects and combines their results into a
single xarray Dataset with proper coordinate handling.

The Observations class is designed to:
- Generate observations across a grid of times and wavelengths
- Efficiently manage memory by sharing system and coronagraph objects
- Combine individual observation results into a unified dataset
- Handle coordinate system merging and attribute tracking
- Provide progress tracking for large observation sets

Key Features:
- Cartesian product generation of time-wavelength combinations
- Automatic coordinate system merging and deduplication
- Distinguishing attribute tracking for dataset organization
- Memory-efficient object sharing between observations
- Progress tracking with tqdm integration

The module uses xarray for structured dataset output and provides comprehensive
metadata tracking for each observation in the batch.
"""

import copy
from itertools import product

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.time import Time
from tqdm import tqdm

import coronagraphoto.util as util


class Observations:
    """Batch observation generator for multiple times and wavelengths.

    This class coordinates the generation of multiple coronagraph observations
    across different observation times and central wavelengths. It efficiently
    manages memory by sharing system and coronagraph objects between individual
    observations while creating unique observation scenarios for each combination.

    The class handles the complete workflow from observation creation through
    result combination, including coordinate system merging, attribute tracking,
    and dataset organization. It provides progress tracking for large observation
    sets and ensures proper metadata preservation.

    Attributes:
        base_observation (Observation):
            Template observation object with default parameters that will be
            copied for each time-wavelength combination.
        base_obs_scenario (ObservingScenario):
            Base observing scenario that will be modified for each observation.
        times (astropy.time.Time):
            Array of observation times to simulate.
        central_wavelengths (astropy.units.Quantity):
            Array of central wavelengths to simulate.
    """

    def __init__(self, base_observation, times, central_wavelengths):
        """Initialize the batch observation generator.

        Creates a new Observations object that will generate observations for
        all combinations of the provided times and central wavelengths. The
        base_observation serves as a template that will be copied and modified
        for each unique combination.

        Args:
            base_observation (Observation):
                Observation object with default parameters that will be used
                as a template for all generated observations. This object will
                be deep copied to avoid modifying the original.
            times (astropy.time.Time):
                Array of observation times at which to make observations.
                Each time will be combined with each wavelength.
            central_wavelengths (astropy.units.Quantity):
                Array of central wavelengths at which to make observations.
                Each wavelength will be combined with each time.

        Note:
            The total number of observations generated will be the product
            of the number of times and the number of wavelengths.
        """
        self.base_observation = copy.deepcopy(base_observation)
        self.base_obs_scenario = self.base_observation.scenario
        self.times = times
        self.central_wavelengths = central_wavelengths

    def run(self, observations=None):
        """Execute all observations and combine results into a unified dataset.

        This method orchestrates the complete batch observation process. It can
        either use pre-created observations or generate them automatically. The
        method handles coordinate system merging, attribute tracking, and dataset
        organization to produce a single xarray Dataset containing all results.

        The method performs the following steps:
        1. Creates observations if not provided
        2. Determines unique dimensions and coordinates across all observations
        3. Identifies distinguishing attributes for dataset organization
        4. Executes each observation with progress tracking
        5. Combines results into a unified dataset with proper metadata

        Args:
            observations (list of Observation, optional):
                Pre-created list of Observation objects. If None, observations
                will be automatically generated using create_observations().

        Returns:
            xarray.Dataset:
                Combined dataset containing all observation results with the
                following features:
                - Unified coordinate system across all observations
                - Distinguishing attributes tracked in dataset attributes
                - Image titles for each observation combination
                - Proper dimensionality handling for time and wavelength axes

        Note:
            The returned dataset includes metadata in the attributes:
            - 'dist_attrs_for_images': List of distinguishing attribute names
            - 'image_titles': Dictionary mapping observation combinations to
              human-readable titles
        """
        if observations is None:
            # Create the observations if none are provided
            observations = self.create_observations()

        # Start by getting the dimensions for the final Dataset
        # unique_dims = set()
        # for _obs in observations:
        #     _, final_dims = _obs.final_coords_and_dims()
        #     unique_dims.update(final_dims)
        # unique_dims = list(unique_dims)
        # To track what's been added
        unique_dims_set = set()
        # To maintain order
        unique_dims = []
        for _obs in observations:
            _, final_dims = _obs.final_coords_and_dims()
            for dim in final_dims:
                if dim not in unique_dims_set:
                    unique_dims_set.add(dim)
                    unique_dims.append(dim)

        # Create a dictionary of sets to store the unique coordinates for each
        # dimension
        unique_coords = {dim: set() for dim in unique_dims}

        # Loop through all scenarios and create the observation then get the unique
        # coordinates for each dimension
        for _obs in observations:
            final_coords, _ = _obs.final_coords_and_dims()
            for coord, dim in zip(final_coords, unique_dims):
                unique_coords[dim].update(coord)

        for dim in unique_dims:
            unique_coords[dim] = np.array(sorted(unique_coords[dim]))

        # Get all the distinguishing attributes, e.g. what we need to add to
        # each observation to tell them apart (start_time, central_wavelength,
        # include_star, etc.)
        (
            dist_attr_by_obs,
            dist_attr_dims,
            dist_attr_values,
        ) = util.find_distinguishing_attributes(*observations)
        unique_dims.extend(dist_attr_dims)
        for dim in dist_attr_dims:
            unique_coords[dim] = dist_attr_values[dim]
        ds_coords = {dim: unique_coords[dim] for dim in unique_dims}
        obs_ds = xr.Dataset(coords=ds_coords)
        obs_ds.attrs["dist_attrs_for_images"] = dist_attr_dims
        obs_ds.attrs["image_titles"] = {}

        # Now loop through all scenarios again and create the observation
        for obs in tqdm(observations, desc="Simulating all observations", position=0):
            obs.create_count_rates()
            _obs_ds = obs.count_photons()
            _dist_attrs = dist_attr_by_obs[obs]
            _dist_attr_values = []
            for dim in dist_attr_dims:
                _dist_val = _dist_attrs[dim]
                if type(_dist_val) == Time:
                    _dist_val = _dist_val.datetime64
                _dist_attr_values.append(_dist_val)
            _dist_attr_values = tuple(_dist_attr_values)

            _obs_title = {}
            # Add the distinguishing attributes to the Dataset
            for attr, val in _dist_attrs.items():
                if attr == "central_wavelength":
                    val_str = f"{val.to(u.nm).value:.0f}"
                    val = val.to(u.nm).value
                elif type(val) == Time:
                    val_str = f"{val.decimalyear:.2f}"
                    val = val.datetime64
                elif type(val) == u.Quantity:
                    unit = val.unit
                    val = val.value
                    attr += f"({unit})"
                    val_str = f"{val:.2f}"
                elif isinstance(val, str):
                    val_str = val
                else:
                    raise NotImplementedError(
                        f"Type {type(val)} not yet implemented in Observations.run()"
                    )
                _obs_title[attr] = val_str
                _obs_ds = _obs_ds.expand_dims({attr: [val]})
            obs_ds = xr.merge([obs_ds, _obs_ds])
            obs_ds.attrs["image_titles"][_dist_attr_values] = _obs_title
        return obs_ds

    def create_observations(self):
        """Create individual Observation objects for all time-wavelength combinations.

        This method generates a list of Observation objects, one for each
        combination of time and central wavelength. It efficiently manages memory
        by sharing the system and coronagraph objects between all observations
        while creating unique observing scenarios for each combination.

        The method performs the following steps:
        1. Generates all time-wavelength combinations using itertools.product
        2. Creates a copy of the base observation for each combination
        3. Shares system and coronagraph objects to conserve memory
        4. Creates unique observing scenarios with specific times and wavelengths
        5. Pre-creates PSF datacube if disk observations are included

        Returns:
            list of Observation:
                List of Observation objects, one for each time-wavelength
                combination. Each observation has:
                - Unique observing scenario with specific time and wavelength
                - Shared system and coronagraph objects (memory efficient)
                - Proper settings loaded for the specific scenario

        Note:
            The total number of observations returned is len(times) * len(central_wavelengths).
            If disk observations are included, the PSF datacube is created once
            and shared across all observations for efficiency.
        """
        observations = []
        all_scenarios = list(product(self.times, self.central_wavelengths))

        # Loop through all scenarios and create the observation then get the unique
        # coordinates for each dimension
        for time, wavelength in all_scenarios:
            obs = copy.copy(self.base_observation)

            # Manually assign the shared "system" and "coronagraph" attributes
            # so that memory doesn't explode for large numbers of observations
            obs.system = self.base_observation.system
            obs.coronagraph = self.base_observation.coronagraph

            # Adjust observation scenario
            obs_scenario = copy.deepcopy(self.base_obs_scenario)
            obs_scenario.start_time = time
            obs_scenario.central_wavelength = wavelength
            obs.load_settings(obs_scenario, obs.settings)

            observations.append(obs)

        if self.base_observation.settings.include_disk:
            # Create the psf datacube and then share it among the observations
            self.base_observation.coronagraph.create_psf_datacube()
            # for obs in observations:
            #     obs.psf_datacube = psfs
            #     obs.has_psf_datacube = True

        return observations
