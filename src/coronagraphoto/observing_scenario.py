import astropy.units as u
from astropy.time import Time

try:
    import tomllib

    has_tomllib = True
except ImportError:
    import toml

    has_tomllib = False


class ObservingScenario:
    def __init__(self, toml_file=None, custom_scenario=None):
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
        self.detector_shape = None
        self.detector_pixel_scale = None

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
            # self.bandpass_model =
            # self.observing_scenario.scenario.get("bandpass_model")
            # self.frac_bandwidth =
            # self.observing_scenario.scenario.get("frac_bandwidth")
        # Check if detector settings are complete
        assert (self.detector_shape is not None) == (
            self.detector_pixel_scale is not None
        ), "Must provide both detector_shape and detector_pixel_scale or neither"
        self.has_detector = self.detector_shape is not None

    def __repr__(self):
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
        # Load the TOML file
        if has_tomllib:
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
            detector = config["detector"]
            if det_shape := detector.get("shape"):
                self.detector_shape = det_shape
            if det_pixel_scale := detector.get("pixel_scale"):
                self.detector_pixel_scale = u.Quantity(**det_pixel_scale) / u.pix
