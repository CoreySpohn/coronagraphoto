import copy
from itertools import product

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.time import Time
from tqdm import tqdm

import coronagraphoto.util as util


class Observations:
    def __init__(self, base_observation, times, central_wavelengths):
        """Class to generate multiple observations.
        Args:
            base_observation (Observation object):
                Observation object with default parameters.
            times (astropy Time object):
                Times at which to make observations.
            central_wavelengths (astropy Quantity object):
                Central wavelengths at which to make observations.

        """
        self.base_observation = copy.deepcopy(base_observation)
        self.base_obs_scenario = self.base_observation.scenario
        self.times = times
        self.central_wavelengths = central_wavelengths

    def run(self, observations=None):
        """Make observations at all times and wavelengths."""

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
            psfs = self.base_observation.coronagraph.get_disk_psfs()
            for obs in observations:
                obs.psf_datacube = psfs
                obs.has_psf_datacube = True

        return observations
