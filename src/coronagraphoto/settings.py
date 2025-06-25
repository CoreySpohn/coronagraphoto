"""Configuration management for coronagraph observation simulations.

This module provides the Settings class for managing configuration parameters
for coronagraph observation simulations. It handles loading settings from TOML
files and provides validation for simulation parameters.

The Settings class controls various aspects of the simulation including:
- Source inclusion flags (star, planets, disk)
- Output format options (frames, sources, spectrum)
- Precision settings (wavelength resolution, time invariance)
- Simulation behavior configuration

Key Features:
- TOML file configuration loading with fallback support
- Custom settings override capability
- Automatic validation of setting names
- Computed properties for derived settings

The module supports both Python 3.11+ (tomllib) and older versions (toml package)
for TOML file parsing.
"""

try:
    import tomllib

    HAS_TOMLLIB = True
except ImportError:
    import toml

    HAS_TOMLLIB = False


class Settings:
    """Configuration manager for coronagraph observation simulations.

    This class manages all configuration parameters for coronagraph observation
    simulations, including source inclusion flags, output format options, and
    precision settings. It provides methods for loading settings from TOML files
    and applying custom overrides.

    The Settings class validates input parameters and provides computed properties
    for derived settings that depend on multiple configuration flags.

    Attributes:
        include_star (bool):
            Whether to include the host star in the simulation.
        include_planets (bool):
            Whether to include exoplanets in the simulation.
        include_disk (bool):
            Whether to include circumstellar disk in the simulation.
        return_frames (bool):
            Whether to return individual observation frames.
        return_sources (bool):
            Whether to return individual source contributions separately.
        return_spectrum (bool):
            Whether to return wavelength-resolved spectra.
        time_invariant_planets (bool):
            Whether planetary properties are time-invariant.
        time_invariant_star (bool):
            Whether stellar properties are time-invariant.
        time_invariant_disk (bool):
            Whether disk properties are time-invariant.
        wavelength_resolved_flux (bool):
            Whether to compute wavelength-resolved flux for sources.
        wavelength_resolved_transmission (bool):
            Whether to compute wavelength-resolved transmission.
        any_wavelength_dependence (bool):
            Computed property indicating if any wavelength-dependent
            calculations are enabled.
    """

    def __init__(self, toml_file=None, custom_settings=None):
        """Initialize Settings with default values and optional configuration.

        Sets up default simulation parameters and optionally loads settings
        from a TOML file and/or applies custom setting overrides. The method
        validates that all custom settings correspond to valid attributes
        before applying them.

        Args:
            toml_file (str or pathlib.Path, optional):
                Path to TOML configuration file. If provided, settings will
                be loaded from this file after applying defaults.
            custom_settings (dict, optional):
                Dictionary of custom setting overrides. Keys must correspond
                to valid Settings attributes. Applied after TOML file loading.

        Raises:
            AttributeError:
                If custom_settings contains keys that don't correspond to
                valid Settings attributes.

        Note:
            Custom settings are applied after TOML file loading, so they
            will override any values loaded from the file.
        """
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
        """Return a string representation of the Settings object.

        Returns:
            str:
                A formatted string showing all settings and their current values.
        """
        attrs = vars(self)
        parts = ["Settings:"]
        for key, value in attrs.items():
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)

    def load_settings(self, toml_file):
        """Load configuration settings from a TOML file.

        This method parses a TOML configuration file and updates the Settings
        object with the values found in the file. The TOML file should be
        structured with sections for different types of settings:

        - [sources]: Controls which astrophysical sources to include
        - [output]: Controls output format and data structure
        - [precision]: Controls simulation precision and wavelength/time resolution

        The method uses the appropriate TOML parser based on Python version
        (tomllib for Python 3.11+, toml package for older versions).

        Args:
            toml_file (str or pathlib.Path):
                Path to the TOML configuration file to load.

        Note:
            Only settings present in the TOML file will be updated. Settings
            not specified in the file retain their current values (defaults
            or previously set values).

        Example TOML structure:
            [sources]
            star = true
            planets = true
            disk = false

            [output]
            spectrum = true
            frames = false
            sources = true

            [precision]
            [precision.wavelength]
            flux = true
            transmission = true

            [precision.time_invariance]
            star = true
            planets = false
            disk = true
        """
        # Load the TOML file
        if HAS_TOMLLIB:
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
