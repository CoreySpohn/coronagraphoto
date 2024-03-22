try:
    import tomllib

    has_tomllib = True
except ImportError:
    import toml

    has_tomllib = False


class Settings:
    def __init__(self, toml_file=None, custom_settings=None):
        # Default settings
        self.include_star = False
        self.include_planets = True
        self.include_disk = False
        self.return_frames = False
        self.return_sources = False
        self.return_spectrum = False
        self.time_invariant_planets = True
        self.time_invariant_star = True
        self.time_invariant_disk = True
        self.wavelength_resolved_flux = False
        self.wavelength_resolved_transmission = False

        if toml_file:
            self.load_settings(toml_file)

        if custom_settings:
            for key, value in custom_settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(f"{key} is not a valid setting.")

        self.any_wavelength_dependence = (
            self.wavelength_resolved_flux or self.wavelength_resolved_transmission
        )

    def __repr__(self):
        attrs = vars(self)
        parts = ["Settings:"]
        for key, value in attrs.items():
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)

    def load_settings(self, toml_file):
        # Load the TOML file
        if has_tomllib:
            with open(toml_file, "rb") as file:
                config = tomllib.load(file)
        else:
            config = toml.load(toml_file)

        # Sources settings
        if "sources" in config:
            # Adding sources settings, which are booleans
            sources = config["sources"]
            if (star := sources.get("star")) is not None:
                self.include_star = star
            if (planets := sources.get("planets")) is not None:
                self.include_planets = planets
            if (disk := sources.get("disk")) is not None:
                self.include_disk = disk

        # Output settings
        if "output" in config:
            # Booleans for output settings
            output = config["output"]
            if (output_spectrum := output.get("spectrum")) is not None:
                self.return_spectrum = output_spectrum
            if (output_frames := output.get("frames")) is not None:
                self.return_frames = output_frames
            if (output_sources := output.get("sources")) is not None:
                self.return_sources = output_sources

        # Precision settings
        if "precision" in config:
            precision = config["precision"]
            if wavelength := precision.get("wavelength"):
                # Adding wavelength resolved settings, which are booleans
                if (flux := wavelength.get("flux")) is not None:
                    self.wavelength_resolved_flux = flux
                if (transmission := wavelength.get("transmission")) is not None:
                    self.wavelength_resolved_transmission = transmission

            if time_invariance := precision.get("time_invariance"):
                # Time invariance settings, which are booleans
                if (star := time_invariance.get("star")) is not None:
                    self.time_invariant_star = star
                if (planets := time_invariance.get("planets")) is not None:
                    self.time_invariant_planets = planets
                if (disk := time_invariance.get("disk")) is not None:
                    self.time_invariant_disk = disk
