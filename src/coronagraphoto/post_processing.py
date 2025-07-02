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
        # Define the components of the processed image
        processed_components = []
        coro_plane = False
        det_plane = False

        # Check for coronagraph-plane data (count rates)
        coro_vars = ["star_rate(coro)", "disk_rate(coro)", "planet_rate(coro)"]
        if all(var in dataset for var in coro_vars):
            coro_plane = True
            star_processed_coro = (
                dataset["star_rate(coro)"] / self.config.star_post_processing_factor
            )
            disk_processed_coro = (
                dataset["disk_rate(coro)"] / self.config.disk_post_processing_factor
            )
            processed_components_coro = [
                star_processed_coro,
                disk_processed_coro,
                dataset["planet_rate(coro)"],
            ]

        # Check for detector-plane data (electron counts)
        det_vars = ["star(det)", "disk(det)", "planet(det)"]
        if all(var in dataset for var in det_vars):
            det_plane = True
            star_processed_det = (
                dataset["star(det)"] / self.config.star_post_processing_factor
            )
            disk_processed_det = (
                dataset["disk(det)"] / self.config.disk_post_processing_factor
            )
            processed_components_det = [
                star_processed_det,
                disk_processed_det,
                dataset["planet(det)"],
            ]

        # Add noise components if they exist
        noise_vars_coro = ["dark_current(coro)", "read_noise(coro)", "cic(coro)"]
        noise_vars_det = ["dark_current(det)", "read_noise(det)", "cic(det)"]

        if any(var in dataset for var in noise_vars_coro) and coro_plane:
            for var in noise_vars_coro:
                if var in dataset:
                    processed_components_coro.append(dataset[var])

        if any(var in dataset for var in noise_vars_det) and det_plane:
            for var in noise_vars_det:
                if var in dataset:
                    processed_components_det.append(dataset[var])

        # Create the processed images by summing components
        if coro_plane:
            dataset["processed_image_rate(coro)"] = sum(processed_components_coro)

        if det_plane:
            dataset["processed_image(det)"] = sum(processed_components_det)

        if not coro_plane and not det_plane:
            raise ValueError(
                "Input dataset for PostProcessing must contain star, disk, and "
                "planet data in at least one plane (coro or det). "
                "Ensure Observation was run with return_sources=True."
            )

        return dataset
