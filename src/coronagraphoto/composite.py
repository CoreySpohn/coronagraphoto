"""Creates a composite image from a list of observations.

This module contains the CompositeObservation class, which manages a collection of
Observation objects. This is useful for creating composite images, such as RGB
images from observations with different bandpasses.
"""

import numpy as np
from astropy.visualization import LinearStretch, MinMaxInterval, make_rgb

from coronagraphoto.observation import Observation


class CompositeObservation:
    """Manages a collection of observations to create a composite image.

    This class takes a list of observing scenarios and runs an observation for
    each one. It can then be used to generate a composite image from the

    results, for example, an RGB image.
    """

    def __init__(
        self,
        coronagraph,
        system,
        observing_scenarios,
        settings,
        speckle_map=None,
        post_processing=None,
    ):
        """Initializes the CompositeObservation object.

        Args:
            coronagraph (Coronagraph):
                The coronagraph model.
            system (System):
                The system to observe.
            observing_scenarios (list[ObservingScenario]):
                A list of observing scenarios.
            settings (Settings):
                The simulation settings.
            speckle_map (SpeckleMap, optional):
                The speckle map object. Defaults to None.
            post_processing (PostProcessing, optional):
                The post-processing object. Defaults to None.
        """
        self.coronagraph = coronagraph
        self.system = system
        self.observing_scenarios = observing_scenarios
        self.settings = settings
        self.speckle_map = speckle_map
        self.post_processing = post_processing

        self.observations = [
            Observation(
                self.coronagraph, self.system, scen, self.settings, self.speckle_map
            )
            for scen in self.observing_scenarios
        ]
        self.datasets = []

    def run(self, time_invariant=False):
        """Runs all observations and stores the resulting datasets."""
        self.datasets = []
        if time_invariant:
            # Create the first observation's count_rates and use it for all others
            self.observations[0].create_count_rates()
            for obs in self.observations[1:]:
                # Copy the count rates from the first observation
                obs.total_count_rate = self.observations[0].total_count_rate.copy()
                if obs.settings.include_star:
                    obs.star_count_rate = self.observations[0].star_count_rate.copy()
                if obs.settings.include_planets:
                    obs.planet_count_rate = self.observations[
                        0
                    ].planet_count_rate.copy()
                if obs.settings.include_disk:
                    obs.disk_count_rate = self.observations[0].disk_count_rate.copy()
                if obs.has_speckle_map:
                    obs.speckle_count_rate = self.observations[
                        0
                    ].speckle_count_rate.copy()

            for obs in self.observations:
                dataset = obs.count_photons()
                if self.post_processing:
                    dataset = self.post_processing.process(dataset)
                self.datasets.append(dataset)
        else:
            for obs in self.observations:
                obs.create_count_rates()
                dataset = obs.count_photons()
                if self.post_processing:
                    dataset = self.post_processing.process(dataset)
                self.datasets.append(dataset)
        return self.datasets

    def get_composite_image(self, image_key="image(det)"):
        """Stacks the images from all observations into a single datacube.

        Args:
            image_key (str):
                The key for the image in the dataset dictionary.
                Defaults to "image(det)". Can also be "processed_image(det)".

        Returns:
            np.ndarray:
                A datacube of shape (n_observations, height, width).
        """
        if not self.datasets:
            raise ValueError("Observations have not been run yet. Call .run() first.")

        images = [d[image_key].squeeze() for d in self.datasets]
        return np.stack(images, axis=0)

    @staticmethod
    def create_rgb_image(
        datacube,
        channel_mapping,
        interval=None,
        stretch=None,
    ):
        """Creates an RGB image from a datacube using astropy.visualization.make_rgb.

        Args:
            datacube (np.ndarray):
                The datacube from get_composite_image(), with shape
                (n_observations, height, width).
            channel_mapping (dict):
                A dict mapping 'r', 'g', 'b' to indices in the datacube's
                first axis. For example, {'r': 0, 'g': 1, 'b': 2}.
            interval (astropy.visualization.BaseInterval, optional):
                The interval object to use for scaling. If None, MinMaxInterval
                is used. Defaults to None.
            stretch (astropy.visualization.BaseStretch, optional):
                The stretch object to use for scaling. If None, LinearStretch
                is used. Defaults to None.

        Returns:
            np.ndarray:
                An RGB image of shape (height, width, 3).
        """
        if datacube.ndim != 3:
            raise ValueError("Datacube must be a 3D array.")

        image_r = datacube[channel_mapping["r"]]
        image_g = datacube[channel_mapping["g"]]
        image_b = datacube[channel_mapping["b"]]

        if interval is None:
            interval = MinMaxInterval()

        if stretch is None:
            stretch = LinearStretch()

        # make_rgb returns an array of floats in the range [0, 1]
        rgb_image = make_rgb(
            image_r, image_g, image_b, interval=interval, stretch=stretch
        )

        return rgb_image
