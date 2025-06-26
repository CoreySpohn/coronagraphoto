"""This is an object to hold all the processing configuration information."""

try:
    import tomllib

    HAS_TOMLLIB = True
except ImportError:
    import toml

    HAS_TOMLLIB = False


class ProcessingConfig:
    """ProcessingConfig holds information required to run post-processing.

    ProcessingConfig has a set of default parameters that can be overwritten, either by
    providing a TOML file with the desired parameters or by providing a dictionary with the
    custom config information. The custom_config values will overwrite anything provided
    in the TOML file.
    """

    def __init__(self, toml_file=None, custom_config=None):
        """Initializes the ProcessingConfig object.

        Args:
            toml_file (str, optional):
                Path to a TOML file with the processing config parameters.
            custom_config (dict, optional):
                Dictionary with custom processing config parameters.
        """
        # Default scenario values
        self.star_post_processing_factor = 1.0
        self.disk_post_processing_factor = 1.0

        # Load the TOML file (if provided)
        if toml_file:
            self.load_toml(toml_file)

        # Update the scenario with the provided custom settings
        if custom_config is not None:
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f"{key} is not a valid observing scenario setting."
                    )

    def __repr__(self):
        """Returns a string representation of the processing config parameters."""
        attrs = vars(self)
        parts = ["ProcessingConfig:"]
        for key, value in attrs.items():
            value_str = repr(value)
            parts.append(f"  {key}: {value_str}")

        return "\n".join(parts)

    def load_toml(self, toml_file):
        """Loads a TOML file and overwrites default parameters with the contents of the file."""
        # Load the TOML file
        if HAS_TOMLLIB:
            with open(toml_file, "rb") as file:
                config = tomllib.load(file)
        else:
            config = toml.load(toml_file)

        # Post-processing settings
        if "post_processing" in config:
            post_processing = config["post_processing"]
            if star_factor := post_processing.get("star_post_processing_factor"):
                self.star_post_processing_factor = star_factor

            if disk_factor := post_processing.get("disk_post_processing_factor"):
                self.disk_post_processing_factor = disk_factor
