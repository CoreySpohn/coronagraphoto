"""This is an object to hold all the post-processing information."""


class PostProcessing:
    """PostProcessing holds information required to run post-processing.

    This class takes a dataset from an Observation and applies post-processing
    effects, such as simulating the subtraction of stellar and disk light.
    """

    def __init__(self, processing_config):
        """Initializes the PostProcessing object.

        Args:
            processing_config (ProcessingConfig):
                The processing configuration object.
        """
        self.config = processing_config

    def process(self, dataset):
        """Apply post-processing to a dataset.

        This method takes a dataset from an observation and applies the
        post-processing steps defined in the ProcessingConfig. It returns the
        dataset with the processed image added as a new data array.

        The main operation is the reduction of star and disk flux to simulate
        the effects of post-processing techniques like reference star subtraction.
        This is done for both the coronagraph and detector planes if they exist.

        Args:
            dataset (xarray.Dataset):
                The dataset from the observation.

        Returns:
            xarray.Dataset:
                The dataset with the post-processing applied.
        """
        # Process coronagraph-plane images
        coro_vars = ["star(coro)", "disk(coro)", "planet(coro)"]
        if all(var in dataset for var in coro_vars):
            dataset["processed_image(coro)"] = (
                dataset["star(coro)"] / self.config.star_post_processing_factor
                + dataset["disk(coro)"] / self.config.disk_post_processing_factor
                + dataset["planet(coro)"]
            )
        else:
            raise ValueError(
                "Input dataset for PostProcessing must contain star(coro), "
                "disk(coro), and planet(coro). Ensure Observation "
                "was run with return_sources=True."
            )

        # Process detector-plane images if they exist
        det_vars = ["star(det)", "disk(det)", "planet(det)"]
        if all(var in dataset for var in det_vars):
            dataset["processed_image(det)"] = (
                dataset["star(det)"] / self.config.star_post_processing_factor
                + dataset["disk(det)"] / self.config.disk_post_processing_factor
                + dataset["planet(det)"]
            )

        return dataset
