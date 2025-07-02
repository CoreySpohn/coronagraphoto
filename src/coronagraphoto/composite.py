"""Creates a composite image from a list of observations.

This module contains the CompositeObservation class, which manages a collection of
Observation objects. This is useful for creating composite images, such as RGB
images from observations with different bandpasses.
"""

import numpy as np
from matplotlib.colors import Normalize

from coronagraphoto.observation import Observation


class CompositeObservation:
    """Manages a collection of observations to create a composite image.

    This class takes a list of observing scenarios and runs an observation for
    each one. It can then be used to generate a composite image from the

    results, for example, an RGB image.
    """

    def __init__(
        self, coronagraph, system, observing_scenarios, settings, post_processing=None
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
            post_processing (PostProcessing, optional):
                The post-processing object. Defaults to None.
        """
        self.coronagraph = coronagraph
        self.system = system
        self.observing_scenarios = observing_scenarios
        self.settings = settings
        self.post_processing = post_processing

        self.observations = [
            Observation(self.coronagraph, self.system, scen, self.settings)
            for scen in self.observing_scenarios
        ]
        self.datasets = []

    def run(self):
        """Runs all observations and stores the resulting datasets."""
        self.datasets = []
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
    def create_rgb_image(datacube, channel_mapping, norm=None):
        """Creates an RGB image from a datacube.

        This method takes a datacube and a mapping from RGB channels to indices
        in the datacube and creates an RGB image. The channels are scaled for
        visualization.

        Args:
            datacube (np.ndarray):
                The datacube from get_composite_image(), with shape
                (n_observations, height, width).
            channel_mapping (dict):
                A dict mapping 'r', 'g', 'b' to indices in the datacube's
                first axis. For example, {'r': 0, 'g': 1, 'b': 2}.
            norm (matplotlib.colors.Normalize, optional):
                A matplotlib Normalize instance to scale the data. If None, a
                linear scaling from the data's min to max is used. Defaults to
                None.

        Returns:
            np.ndarray: An RGB image of shape (height, width, 3).
        """
        if datacube.ndim != 3:
            raise ValueError("Datacube must be a 3D array.")

        h, w = datacube.shape[1], datacube.shape[2]
        rgb_image = np.zeros((h, w, 3))

        if norm is None:
            # Create a default linear normalization
            # Using only the channels that are being mapped
            mapped_channels = [
                channel_mapping[c] for c in ["r", "g", "b"] if c in channel_mapping
            ]
            vmin = datacube[mapped_channels].min()
            vmax = datacube[mapped_channels].max()
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        if "r" in channel_mapping:
            rgb_image[..., 0] = norm(datacube[channel_mapping["r"]])
        if "g" in channel_mapping:
            rgb_image[..., 1] = norm(datacube[channel_mapping["g"]])
        if "b" in channel_mapping:
            rgb_image[..., 2] = norm(datacube[channel_mapping["b"]])

        return rgb_image
