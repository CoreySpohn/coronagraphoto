"""This is an object to hold all the observing scenario information."""

import astropy.units as u
from astropy.time import Time

try:
    import tomllib

    HAS_TOMLLIB = True
except ImportError:
    import toml  # type: ignore

    HAS_TOMLLIB = False

from coronagraphoto.detector import Detector


class ObservingScenario:
    """ObservingScenario holds information required to run an observation.

    ObservingScenario has a set of default parameters that can be overwritten, either by
    providing a TOML file with the desired parameters or by providing a dictionary with the
    custom scenario information. The custom_scenario values will overwrite anything provided
    in the TOML file.
    """

    def __init__(self, toml_file=None, custom_scenario=None):
        """Initializes the ObservingScenario object.

        Args:
            toml_file (str, optional):
                Path to a TOML file with the observing scenario parameters.
            custom_scenario (dict, optional):
                Dictionary with custom observing scenario parameters.
        """
        # Default scenario values
        self.diameter = 1 * u.m
        self.central_wavelength = 500 * u.nm
        self.start_time = Time(2000, format="decimalyear")
        self.exposure_time = 1 * u.d
        self.frame_time = 1 * u.hr
        self.bandpass = None
        # self.bandpass = SpectralElement(
        #     "Gaussian1D",
        #     mean=self.central_wavelength,
        #     stddev=0.2 * self.central_wavelength / np.sqrt(2 * np.pi),
        # )
        self.spectral_resolution = 100
        self.detector = None

        # Load the TOML file (if provided)
        if toml_file:
            self.load_toml(toml_file)

        # Update the scenario with the provided custom settings
        if custom_scenario is not None:
            for key, value in custom_scenario.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f"{key} is not a valid observing scenario setting."
                    )

        # Load bandpass
        if self.bandpass is None:
            raise ValueError("Must provide a bandpass model currently")

        # Check if detector settings are complete
        self.has_detector = self.detector is not None

        self.frame_time_s = self.frame_time.to_value(u.s)
        self.exposure_time_s = self.exposure_time.to_value(u.s)

    def __repr__(self):
        """Returns a string representation of the observing scenario parameters."""
        attrs = vars(self)
        parts = ["ObservingScenario:"]
        for key, value in attrs.items():
            if hasattr(value, "unit"):
                # Currently just use astropy print
                value_str = str(value)
            elif isinstance(value, Time):
                # Special handling for astropy Time objects
                value_str = str(value.datetime)
            else:
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

        # General settings
        if "general" in config:
            general = config["general"]
            if diameter := general.get("diameter"):
                self.diameter = u.Quantity(**diameter)

            if exposure_time := general.get("exposure_time"):
                self.exposure_time = u.Quantity(**exposure_time)

            if frame_time := general.get("frame_time"):
                self.frame_time = u.Quantity(**frame_time)

            if start_time := general.get("start_time"):
                self.start_time = Time(**start_time)

        # Spectral information
        if "spectral_information" in config:
            # Adding spectral information
            spectral_information = config["spectral_information"]
            if central_wavelength := spectral_information.get("central_wavelength"):
                self.central_wavelength = u.Quantity(**central_wavelength)

            if spectral_resolution := spectral_information.get("spectral_resolution"):
                # Should be an integer
                self.spectral_resolution = spectral_resolution

        # Detector settings
        if "detector" in config:
            detector_config = config["detector"]
            # Create a dictionary of arguments for the Detector class
            detector_args = {}
            if shape := detector_config.get("shape"):
                detector_args["shape"] = shape
            if pixel_scale := detector_config.get("pixel_scale"):
                detector_args["pixel_scale"] = u.Quantity(**pixel_scale) / u.pix
            if quantum_efficiency := detector_config.get("quantum_efficiency"):
                detector_args["quantum_efficiency"] = quantum_efficiency
            if dark_current_rate := detector_config.get("dark_current_rate"):
                detector_args["dark_current_rate"] = u.Quantity(**dark_current_rate)
            if read_noise := detector_config.get("read_noise"):
                detector_args["read_noise"] = u.Quantity(**read_noise)
            if cic_rate := detector_config.get("cic_rate"):
                detector_args["cic_rate"] = u.Quantity(**cic_rate)

            # Check if all necessary detector arguments are present
            if "shape" in detector_args and "pixel_scale" in detector_args:
                self.detector = Detector(**detector_args)
            elif detector_args:
                raise ValueError(
                    "Detector config must include at least shape and pixel_scale."
                )
