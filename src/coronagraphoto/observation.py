"""Coronagraph observation simulation module.

This module provides the core functionality for simulating coronagraph observations
of exoplanetary systems. It handles the complete observation pipeline from count
rate generation through photon noise simulation to final dataset creation.

The main class, Observation, coordinates the simulation of various astrophysical
sources (host stars, exoplanets, circumstellar disks) as they would appear through
a coronagraph instrument. It accounts for:

- Wavelength-dependent and time-dependent source properties
- Coronagraph transmission and PSF effects
- Orbital dynamics for planetary motion
- Poisson photon noise simulation
- Multiple output formats (frames vs. integrated, spectra vs. broadband)

Key Features:
- Support for both time-invariant and time-varying simulations
- Wavelength-resolved or broadband observations
- Individual source tracking or combined imaging
- Coronagraph and detector pixel scale handling
- Comprehensive validation and error checking

The module uses JAX for efficient numerical computations, Astropy for astronomical
units and coordinate handling, and xarray for structured dataset output.
"""

import copy
from pathlib import Path

import astropy.units as u
import astropy.units.equivalencies as equiv
import jax
import jax.numpy as jnp
import lod_unit  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.stats import SigmaClip
from astropy.time import Time
from exoverses.util import misc
from matplotlib.colors import LogNorm
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from scipy.ndimage import gaussian_filter, rotate, shift, zoom
from tqdm import tqdm

from coronagraphoto import util

from .logger import logger


class Observation:
    """Simulate coronagraph observations of exoplanetary systems.

    This class orchestrates the complete simulation pipeline for coronagraph
    observations, from initial configuration through count rate generation to
    final photon-noise limited dataset creation. It coordinates multiple
    astrophysical sources and instrument effects to produce realistic
    observation simulations.

    The simulation workflow follows these main steps:
    1. Configuration and validation (load_settings)
    2. Count rate generation for each source (create_count_rates)
    3. Photon noise simulation (count_photons)
    4. Dataset formatting with proper coordinates and dimensions

    Key Capabilities:
    - Multi-source simulations (star, planets, circumstellar disk)
    - Time-dependent orbital dynamics for planets
    - Wavelength-resolved or broadband observations
    - Coronagraph PSF and transmission modeling
    - Poisson photon noise simulation
    - Flexible output formats (individual sources, frames, spectra)
    - Automatic coordinate system handling (coronagraph and detector pixels)

    Attributes:
        coronagraph (Coronagraph):
            Coronagraph instrument model with PSF and transmission data.
        system (ExovistaSystem):
            Stellar system containing star, planets, and disk components.
        scenario (ObservingScenario):
            Observation parameters including timing, wavelengths, and detector setup.
        settings (Settings):
            Simulation configuration flags and output options.
        nframes (int):
            Number of observation frames.
        frame_start_times (astropy.time.Time):
            Start times for each observation frame.
        spectral_wavelength_grid (astropy.units.Quantity):
            Wavelength grid for spectral simulations (if wavelength-dependent).
        illuminated_area (astropy.units.Quantity):
            Effective collecting area of the telescope.
        total_count_rate (astropy.units.Quantity):
            Combined count rate from all included sources.
    """

    def __init__(self, coronagraph, system, observing_scenario, settings):
        """Initialize an Observation simulation.

        Sets up the observation by storing input objects, configuring simulation
        settings, and creating the results directory structure. The initialization
        automatically calls load_settings to validate inputs and configure the
        observation parameters.

        Args:
            coronagraph (Coronagraph):
                Coronagraph instrument model containing PSF data, transmission
                properties, and pixel scale information.
            system (ExovistaSystem):
                Stellar system model containing star, planet, and disk components
                with their physical and orbital properties.
            observing_scenario (ObservingScenario):
                Observation configuration including exposure time, wavelength
                bandpass, detector setup, and timing parameters.
            settings (Settings):
                Simulation settings controlling source inclusion, wavelength
                dependence, output formats, and time dependence options.
        """
        self.coronagraph = coronagraph
        self.coronagraph_name = coronagraph.name
        self.system = system
        self.system_name = system.star.name

        self.load_settings(observing_scenario, settings)

        # Create save directory
        self.save_dir = Path(
            "results", system.file.stem, coronagraph.yip_path.parts[-1]
        )
        self.spec_flux_density_unit = u.photon / (u.m**2 * u.s * u.nm)
        self.rad_per_pix_unit = u.rad / u.pix
        self.photon_per_sec_unit = u.ph / u.s

    def load_settings(self, observing_scenario, settings):
        """Load and validate observation settings and configure the observation.

        This method performs the initial setup for the observation by:
        1. Storing the observing scenario and settings objects
        2. Calculating frame information (number and timing)
        3. Validating input consistency for spectral settings
        4. Creating wavelength grids and bandwidths (if wavelength-dependent)
        5. Setting up transmission arrays (wavelength-dependent or single value)
        6. Computing the effective illuminated area of the telescope

        The method validates that spectral resolution is provided when returning
        spectra, and ensures wavelength dependence settings are consistent.

        Args:
            observing_scenario (ObservingScenario):
                Object containing observation parameters including bandpass,
                exposure time, detector configuration, and timing information.
            settings (Settings):
                Object containing simulation settings including source inclusion
                flags, wavelength dependence options, and output format choices.
        """
        self.scenario = observing_scenario
        self.settings = settings

        # Load observing scenario flags
        self.nframes, self.frame_start_times = self.calc_frame_info()

        # Check inputs
        if self.settings.return_spectrum:
            assert self.scenario.spectral_resolution is not None, (
                "Must provide a scenario.spectral_resolution if "
                "settings.return_spectrum is True"
            )
            assert self.settings.any_wavelength_dependence, (
                "One or both of wavelength_resolved_flux and "
                "wavelength_resolved_transmission must be True "
                "if settings.return_spectrum is True"
            )
        # Create the wavelength grid and bandwidth
        if self.settings.any_wavelength_dependence:
            logger.info("Creating wavelength grid")
            (
                self.spectral_wavelength_grid,
                self.spectral_bandwidths,
            ) = util.gen_wavelength_grid(
                self.scenario.bandpass, self.scenario.spectral_resolution
            )
        else:
            self.full_bandwidth = (
                self.scenario.bandpass.waveset[-1] - self.scenario.bandpass.waveset[0]
            )

        # Create the transmission array (or single value)
        if self.settings.wavelength_resolved_transmission:
            self.spectral_transmission = self.scenario.bandpass(
                self.spectral_wavelength_grid
            ).value
        else:
            self.central_transmission = self.scenario.bandpass(
                self.scenario.central_wavelength
            ).value

        # Solve for illuminated area
        self.illuminated_area = (
            np.pi
            * self.scenario.diameter**2
            / 4.0
            * (1.0 - self.coronagraph.frac_obscured)
        )

    def create_count_rates(self):
        """Create the images at the wavelengths and times."""
        logger.info("Creating count rates")

        if self.settings.any_wavelength_dependence:
            nwave = len(self.spectral_wavelength_grid)
        else:
            nwave = 1
        # if not self.prop_during_exposure:
        npix = self.coronagraph.npixels
        shape = [self.nframes, nwave, npix, npix]

        self.total_count_rate = np.zeros(tuple(shape)) * u.ph / u.s

        base_count_rate_arr = np.zeros_like(self.total_count_rate.value) * u.ph / u.s
        if self.settings.include_star:
            logger.info("Creating star count rate")
            self.star_count_rate = self.generic_count_rate_logic(
                self.gen_star_count_rate,
                base_count_rate_arr,
                time_invariant=self.settings.time_invariant_star,
            )
            self.total_count_rate += self.star_count_rate
        else:
            logger.info("Not including star")

        if self.settings.include_planets:
            logger.info("Creating planets count rate")
            self.planet_count_rate = self.generic_count_rate_logic(
                self.gen_planet_count_rate,
                base_count_rate_arr,
                time_invariant=self.settings.time_invariant_planets,
            )
            self.total_count_rate += self.planet_count_rate
        else:
            logger.info("Not including planets")

        if self.settings.include_disk:
            if not self.coronagraph.has_psf_datacube:
                self.coronagraph.create_psf_datacube()
            logger.info("Creating disk count rate")
            self.disk_count_rate = self.generic_count_rate_logic(
                self.gen_disk_count_rate,
                base_count_rate_arr,
                time_invariant=self.settings.time_invariant_disk,
            )
            self.total_count_rate += self.disk_count_rate
        else:
            logger.info("Not including disk")

    def generic_count_rate_logic(
        self, count_rate_function, object_count_rate, *args, time_invariant=False
    ):
        """Compute count rates for star, planets, and disk over time and wavelength.

        This function handles the logic for computing count rates. It handles
        both time-invariant and time-varying scenarios, as well as
        wavelength-dependent and wavelength-independent calculations. The
        function applies the appropriate transmission and bandwidth corrections
        to the count rates and returns the updated count rate array.

        Args:
            count_rate_function (callable):
                Function that computes count rate.
            object_count_rate (numpy.ndarray):
                Array for computed count rates. Shape: (nframes, nwavelengths,
                npixels, npixels).
            *args:
                Additional arguments to pass to the count_rate_function.
            time_invariant (bool):
                Flag indicating if the count rate is time-invariant.

        Returns:
            numpy.ndarray:
                Updated object_count_rate with computed count rates.
        """
        # Copy the array to avoid modifying the input
        object_count_rate = copy.deepcopy(object_count_rate)

        # Determine the dimensions
        _, nlam, *_ = object_count_rate.shape

        # Gross logic to determine the minimal set of values needed to generate
        # the count rates at

        # Set up whether we need to tile to other wavelengths
        tile_lam = (
            self.settings.wavelength_resolved_transmission
            and not self.settings.wavelength_resolved_flux
        )

        # Set up wavelength and bandwidth arrays
        if self.settings.wavelength_resolved_flux:
            lams = self.spectral_wavelength_grid
            bws = self.spectral_bandwidths
        else:
            lams = [self.scenario.central_wavelength]
            if self.settings.wavelength_resolved_transmission:
                # bandwidth at the central wavelength
                bws = np.repeat(self.spectral_bandwidths[nlam // 2], nlam)
            else:
                bws = [self.full_bandwidth]

        # Create transmission array
        if self.settings.wavelength_resolved_transmission:
            transmissions = self.spectral_transmission
        else:
            # This is a single value, so we can just repeat it
            transmissions = np.repeat(self.central_transmission, nlam)

        if time_invariant:
            frame_times = [self.scenario.start_time]
        else:
            frame_times = self.frame_start_times

        for frame_ind, frame_time in enumerate(
            tqdm(frame_times, desc="Generating count per frame", delay=0.5)
        ):
            for lam_ind, (lam, bw) in enumerate(
                tqdm(
                    zip(lams, bws),
                    desc="Generating count per wavelength",
                    total=len(bws),
                    delay=0.5,
                    position=1,
                )
            ):
                # Calculate the count rate (npix, npix)
                base_count_rate = count_rate_function(lam, frame_time, bw, *args)

                if tile_lam:
                    # Apply transmission to separate the counts by wavelength
                    # (nlam, npixels, npixels)
                    object_count_rate[frame_ind, :] = (
                        transmissions[:, None, None] * base_count_rate
                    )

                else:
                    # Apply transmission to the current lam count rate
                    # (npix, npix)
                    trans_applied_rate = transmissions[lam_ind] * base_count_rate
                    object_count_rate[frame_ind, lam_ind] = trans_applied_rate

            if time_invariant:
                # No change between frames, so apply this to all frames
                object_count_rate[:] = np.repeat(
                    object_count_rate[0][None, ...], self.nframes, axis=0
                )
        return object_count_rate

    def gen_star_count_rate(self, wavelength, time, bandwidth):
        """Generate the star count rate in photons per second.

        This method computes the stellar count rate by:
        1. Converting the star's angular diameter to lambda/D units
        2. Retrieving the stellar intensity map from the coronagraph model
        3. Calculating the star's spectral flux density at the given wavelength and time
        4. Converting flux density to photon count rate units
        5. Multiplying the intensity map by the photon flux to get spatial count rates

        The resulting count rate represents the star's contribution before any
        coronagraph transmission effects are applied.

        Args:
            wavelength (astropy.units.Quantity):
                Observation wavelength.
            time (astropy.time.Time):
                Observation time (for potential stellar variability).
            bandwidth (astropy.units.Quantity):
                Spectral bandwidth for flux integration.

        Returns:
            numpy.ndarray:
                Star count rate image in units of photons per second, with shape
                (npixels, npixels) matching the coronagraph model dimensions.

        Note:
            This count rate is calculated WITHOUT any coronagraph transmission
            effects. Transmission is applied later in the processing pipeline.
        """
        # Compute star count rate in lambda/D
        stellar_diam_lod = self.system.star.angular_diameter.to(
            u.lod, equiv.lod(wavelength, self.scenario.diameter)
        )

        # Get the intensity map I(x,y) at the stellar diameters
        stellar_intens = self.coronagraph.stellar_intens(stellar_diam_lod).T

        # Calculate the star flux density
        star_flux_density = self.system.star.spec_flux_density(wavelength, time).to(
            self.spec_flux_density_unit,
            equivalencies=u.spectral_density(wavelength),
        )

        # Multiply by the count rate term (A*dLambda*T)
        flux_term = (star_flux_density * self.illuminated_area * bandwidth).decompose()

        # Compute star count rate in each pixel
        count_rate = np.multiply(stellar_intens, flux_term).T
        return count_rate

    def gen_planet_count_rate(self, wavelength, time, bandwidth):
        """Generate the planet count rate in photons per second.

        This method computes the planet count rate by:
        1. Propagating planetary orbits to the observation time using n-body dynamics
        2. Converting orbital positions to pixel coordinates in the coronagraph frame
        3. Calculating separations and position angles relative to the star
        4. Computing spectral flux density for each planet at the given wavelength
        5. Converting flux to photon count rates
        6. Applying the appropriate off-axis PSF for each planet's position
        7. Summing contributions from all planets in the system

        NOTE: For 1D coronagraph models, planets beyond the maximum offset
        range are excluded from the calculation (flux set to zero) because it
        results in a significantly clipped PSF.

        Args:
            wavelength (astropy.units.Quantity):
                Observation wavelength.
            time (astropy.time.Time):
                Observation time for orbital propagation.
            bandwidth (astropy.units.Quantity):
                Spectral bandwidth for flux integration.

        Returns:
            astropy.units.Quantity:
                Planet count rate image in units of photons per second, with shape
                (npixels, npixels) matching the coronagraph model dimensions.
                Contains the combined contribution from all planets in the system.

        Note:
            Orbital propagation uses n-body dynamics in heliocentric sky frame.
            Off-axis PSFs are interpolated based on each planet's separation and
            position angle relative to the star.
        """
        # Compute planet separations and position angles.
        prop_kwargs = {
            "prop": "nbody",
            "ref_frame": "helio-sky",
        }
        orbit_dataset = self.system.propagate(time, **prop_kwargs)
        xystar = np.array([self.coronagraph.npixels / 2] * 2) * u.pix
        pixscale = (self.coronagraph.pixel_scale * u.pix).to(
            u.arcsec, equiv.lod(wavelength, self.scenario.diameter)
        ) / u.pix
        orbit_dataset = misc.add_units(
            orbit_dataset,
            u.pixel,
            distance=self.system.star.dist,
            pixel_scale=pixscale,
            star_pixel=xystar,
        )
        pixel_data = orbit_dataset.sel(object="planet", **prop_kwargs)[
            ["x(pix)", "y(pix)"]
        ]
        xyplanet = (
            np.stack(
                [pixel_data["x(pix)"].values, pixel_data["y(pix)"].values], axis=-1
            )
            * u.pix
        )
        planet_xy_separations = (xyplanet - xystar) * pixscale

        # plan_offs
        planet_alphas = np.sqrt(np.sum(planet_xy_separations**2, axis=1))
        planet_angles = np.arctan2(
            planet_xy_separations[:, 1], planet_xy_separations[:, 0]
        )
        planet_alphas_lod = planet_alphas.to(
            u.lod, equiv.lod(wavelength, self.scenario.diameter)
        )

        # Compute planet flux.
        planet_flux_density = np.zeros(len(self.system.planets)) * u.Jy
        for i, planet in enumerate(self.system.planets):
            if self.coronagraph.offax.type == "1d":
                # Check if the planet's PSF center is within the range of separations
                # that the 1d coronagraph provided, otherwise leave the flux as 0
                if planet_alphas_lod[i] > self.coronagraph.offax.max_offset_in_image:
                    continue
            planet_flux_density[i] = planet.spec_flux_density(wavelength, time)

        planet_photon_flux = (
            planet_flux_density.to(
                self.spec_flux_density_unit,
                equivalencies=u.spectral_density(wavelength),
            )
            * self.illuminated_area
            * bandwidth
        ).decompose()

        # Multiply the PSF by the planet flux
        planet_count_rate = (
            np.zeros((self.coronagraph.npixels, self.coronagraph.npixels))
            * planet_photon_flux.unit
        )
        for i, (x, y) in enumerate(planet_xy_separations):
            psf = self.coronagraph.offax(x, y, lam=wavelength, D=self.scenario.diameter)
            planet_count_rate += planet_photon_flux[i] * psf

        return planet_count_rate

    def gen_disk_count_rate(self, wavelength, time, bandwidth):
        """Generate the disk count rate in photons per second.

        This method processes the disk flux density from the system model and converts
        it to a count rate image by:
        1. Retrieving disk spectral flux density at the given wavelength and time
        2. Converting flux density to photon count rate units
        3. Scaling the disk image to match coronagraph pixel scale (lambda/D)
        4. Centering and cropping the disk to coronagraph dimensions
        5. Convolving with the coronagraph PSF datacube to account for instrument response

        The scaling process uses logarithmic interpolation to preserve flux conservation
        and avoid negative values during a scipy.ndimage.zoom operation.

        Args:
            wavelength (astropy.units.Quantity):
                Observation wavelength.
            time (astropy.time.Time):
                Observation time (in case there are time-dependent disk properties).
            bandwidth (astropy.units.Quantity):
                Spectral bandwidth for flux integration.

        Returns:
            astropy.units.Quantity:
                Disk count rate image in units of photons per second, with shape
                (npixels, npixels) matching the coronagraph model dimensions.

        Note:
            This method requires that the coronagraph has a PSF datacube created
            (coronagraph.has_psf_datacube should be True).
        """
        disk_image = self.system.disk.spec_flux_density(wavelength, time)

        # This is the factor to scale the disk image, from exovista, to the
        # coronagraph model size since they do not necessarily have the same
        # pixel scale
        zoom_factor = (
            (u.pixel * self.system.star.pixel_scale.to(self.rad_per_pix_unit)).to(
                u.lod, equiv.lod(wavelength, self.scenario.diameter)
            )
            / self.coronagraph.pixel_scale
        ).value
        # This is the photons per second
        disk_image_photons = (
            disk_image.to(
                self.spec_flux_density_unit,
                equivalencies=u.spectral_density(wavelength),
            )
            * self.illuminated_area
            * bandwidth
        ).value
        scaled_disk = util.zoom_conserve_flux(disk_image_photons, zoom_factor)

        # Center disk so that (img_pixels-1)/2 is center.
        disk_pixels_is_even = scaled_disk.shape[0] % 2 == 0
        coro_pixels_is_even = self.coronagraph.npixels % 2 == 0
        pad_value = scaled_disk.min()
        if disk_pixels_is_even != coro_pixels_is_even:
            scaled_disk = np.pad(
                scaled_disk,
                ((0, 1), (0, 1)),
                mode="edge",
            )
            # interpolate in log-space to avoid negative values
            scaled_disk = np.exp(shift(np.log(scaled_disk), (0.5, 0.5), order=5))
            scaled_disk = scaled_disk[1:-1, 1:-1]

        # Crop disk to coronagraph model size.
        if scaled_disk.shape[0] == self.coronagraph.npixels:
            pass
        elif scaled_disk.shape[0] > self.coronagraph.npixels:
            # Crop the disk down to the coronagraph model size
            nn = (scaled_disk.shape[0] - self.coronagraph.npixels) // 2
            scaled_disk = scaled_disk[nn:-nn, nn:-nn]
        else:
            # Pad the disk up to the coronagraph model size
            nn = (self.coronagraph.npixels - scaled_disk.shape[0]) // 2
            # Calculate the number of missing pixels in the image
            disk_pix = scaled_disk.shape[0] ** 2
            coro_pix = self.coronagraph.npixels**2
            missing_pix = coro_pix - disk_pix
            frac_missing_pix = missing_pix / coro_pix

            # Compare linear_ramp vs physically motivated padding
            min_val = np.log(scaled_disk.min() / 1e8)

            # Original linear_ramp method
            scaled_disk = np.exp(
                np.pad(
                    np.log(scaled_disk),
                    ((nn, nn), (nn, nn)),
                    mode="linear_ramp",
                    end_values=(min_val, min_val),
                )
            )

            if frac_missing_pix > 0.01:
                # Some useful debug information for fixing this issue
                # Get the size of the arrays in lambda/D
                coro_lam_d_pix = self.coronagraph.pixel_scale.value
                exo_lam_d_pix = zoom_factor * self.coronagraph.pixel_scale.value
                # Calculate the number of pixels required in ExoVista to match the coronagraph model's size
                # at the given wavelength and diameter
                required_exo_pix = (
                    coro_lam_d_pix / exo_lam_d_pix * self.coronagraph.npixels
                )
                # Calculate the pixel scale (in mas) required in ExoVista to match the coronagraph model's size
                # at the given wavelength and diameter
                lam_d_width = coro_lam_d_pix * self.coronagraph.npixels * u.lod
                exo_npix = disk_image.shape[0]
                exo_pix_scale_lam_d = lam_d_width / exo_npix
                exo_pix_scale_mas = exo_pix_scale_lam_d.to(
                    u.mas, equiv.lod(wavelength, self.scenario.diameter)
                )

                # Calculate intermediate combinations for user convenience
                current_pix_scale_mas = self.system.disk.pixel_scale.to_value(
                    u.mas / u.pix
                )

                # Calculate the required field of view
                required_field_of_view_mas = exo_pix_scale_mas.value * exo_npix

                # Create table of options
                table_header = "\n\nExoVista Settings options to avoid this issue:"
                table_header += "\n  pixscale (arcsec) | npix | Description"
                table_header += "\n  ------------------|------|------------------"

                # Convert mas to arcsec for ExoVista input format
                current_pix_scale_arcsec = current_pix_scale_mas / 1000.0
                exo_pix_scale_arcsec = exo_pix_scale_mas.value / 1000.0

                # Current configuration
                current_field_of_view_mas = current_pix_scale_mas * exo_npix
                table_rows = f"\n  {current_pix_scale_arcsec:>17.5f} | {exo_npix:>4} | Current (insufficient)"

                # Optimal solutions
                table_rows += f"\n  {exo_pix_scale_arcsec:>17.5f} | {exo_npix:>4} | Just increase pixscale"
                table_rows += f"\n  {current_pix_scale_arcsec:>17.5f} | {required_exo_pix:>4.0f} | Just increase npix"

                # Factor-based combinations - maintain required field of view
                factors = [1.5, 2.0, 3.0]
                for factor in factors:
                    # Option 1: Increase npix by factor from current, calculate required pixel scale
                    new_npix = int(exo_npix * factor)
                    new_pix_scale_mas = required_field_of_view_mas / new_npix
                    new_pix_scale_arcsec = new_pix_scale_mas / 1000.0
                    table_rows += f"\n  {new_pix_scale_arcsec:>17.5f} | {new_npix:>4} | {factor}x npix from current ({exo_npix})"

                    # Option 2: Increase pixel scale by factor from current, calculate required npix
                    new_pix_scale_2_mas = current_pix_scale_mas * factor
                    new_pix_scale_2_arcsec = new_pix_scale_2_mas / 1000.0
                    new_npix_2 = int(required_field_of_view_mas / new_pix_scale_2_mas)
                    table_rows += f"\n  {new_pix_scale_2_arcsec:>17.5f} | {new_npix_2:>4} | {factor}x pixscale from current ({current_pix_scale_arcsec:.5f})"

                combo_text = table_header + table_rows

                logger.warning(
                    "\n****************************************************"
                    f"\nMISSING INFORMATION FOR {100 * frac_missing_pix:.2f}% OF THE PIXELS IN THE DISK IMAGE."
                    f"\nThe ExoVista disk is smaller than coronagraph model at lambda={wavelength.to_value(u.nm):.0f} nm, "
                    f"and D={self.scenario.diameter.to_value(u.m):.0f} m by {nn} pixels on each side. "
                    f"\nThe current solution is padding the disk with {nn} pixels on each side and filling the values "
                    "with an exponential decay of the edge values. "
                    "\nI highly recommend increasing the pixel scale (pixscale) or number of pixels (npix) in "
                    "ExoVista to avoid this kind of naive interpolation."
                    f"{combo_text}"
                    "\n****************************************************"
                )

        scaled_disk = np.ascontiguousarray(scaled_disk)

        # Convolve with the PSF datacube
        count_rate = (
            compute_disk_image(scaled_disk, self.coronagraph.psf_datacube)
            << self.photon_per_sec_unit
        )
        return count_rate

    def count_photons(self):
        """Simulate photon collection as a Poisson process and create observation dataset.

        This method converts count rates into actual photon counts by sampling from
        Poisson distributions. It handles different simulation modes based on settings:

        - If settings.return_sources is True, each source (star, planets, disk) is
          simulated separately and added to create the total image
        - If False, the total count rate is used directly for efficiency

        The method creates an xarray Dataset containing the simulated observations
        with appropriate coordinates and dimensions. Conditional processing is applied
        based on settings for spectrum and frame return options.

        Returns:
            xarray.Dataset:
                Dataset containing the simulated photon counts with the following
                possible data variables:

                - 'img(coro)': Total image counts in coronagraph pixels
                - 'img(det)': Total image counts in detector pixels (if detector defined)
                - 'star(coro)', 'planet(coro)', 'disk(coro)': Individual source counts
                  in coronagraph pixels (if settings.return_sources is True)
                - 'star(det)', 'planet(det)', 'disk(det)': Individual source counts
                  in detector pixels (if settings.return_sources is True and detector defined)

                Dimensions depend on settings:
                - 'time': Present if settings.return_frames is True
                - 'spectral_wavelength(nm)': Present if settings.return_spectrum is True
                - 'xpix(coro)', 'ypix(coro)': Coronagraph pixel coordinates
                - 'xpix(det)', 'ypix(det)': Detector pixel coordinates (if applicable)
        """
        logger.info("Creating images")

        coro_image_shape = self.get_coro_image_shape()

        # Create the arrays to store the frame counts for each source
        # if we're returning them
        if self.settings.return_sources:
            frame_counts = np.zeros(coro_image_shape)
            if self.settings.include_star:
                expected_star_photons_per_frame = (
                    self.star_count_rate.to_value(self.photon_per_sec_unit)
                    * self.scenario.frame_time_s
                )
                star_frame_counts = np.random.poisson(expected_star_photons_per_frame)
                frame_counts += star_frame_counts
            if self.settings.include_planets:
                expected_planet_photons_per_frame = (
                    self.planet_count_rate.to_value(self.photon_per_sec_unit)
                    * self.scenario.frame_time_s
                )
                planet_frame_counts = np.random.poisson(
                    expected_planet_photons_per_frame
                )
                frame_counts += planet_frame_counts
            if self.settings.include_disk:
                expected_disk_photons_per_frame = (
                    self.disk_count_rate.to_value(self.photon_per_sec_unit)
                    * self.scenario.frame_time_s
                )
                disk_frame_counts = np.random.poisson(expected_disk_photons_per_frame)
                frame_counts += disk_frame_counts
        else:
            # Calculate the expected number of photons per frame
            expected_photons_per_frame = (
                self.total_count_rate.to_value(self.photon_per_sec_unit)
                * self.scenario.frame_time_s
            )
            frame_counts = np.random.poisson(expected_photons_per_frame)

        coro_coords, coro_dims, det_coords, det_dims = self.coro_det_coords_and_dims()
        args = (coro_coords, coro_dims, det_coords, det_dims)

        coords_dict = {dim: coro_coords[i] for i, dim in enumerate(coro_dims)}
        obs_ds = xr.Dataset(coords=coords_dict)
        obs_ds = self.add_source_to_dataset(frame_counts, "img", obs_ds, *args)

        if self.settings.return_sources:
            if self.settings.include_star:
                obs_ds = self.add_source_to_dataset(
                    star_frame_counts, "star", obs_ds, *args
                )
            if self.settings.include_planets:
                obs_ds = self.add_source_to_dataset(
                    planet_frame_counts, "planet", obs_ds, *args
                )
            if self.settings.include_disk:
                obs_ds = self.add_source_to_dataset(
                    disk_frame_counts, "disk", obs_ds, *args
                )
        if not self.settings.return_spectrum:
            # Sum over the wavelength axis if we're not returning the spectrum
            # but we did calculate it
            obs_ds = obs_ds.sum(dim="spectral_wavelength(nm)")

        if not self.settings.return_frames:
            # Sum over the time axis if we're not returning the frames
            obs_ds = obs_ds.sum(dim="time")

        return obs_ds

    def add_source_to_dataset(
        self, coro_counts, source_name, ds, coro_coords, coro_dims, det_coords, det_dims
    ):
        """Create and add source DataArrays for coronagraph and detector.

        This method creates xarray DataArrays for the given source counts at both
        coronagraph and detector scales, then merges them into the provided dataset.
        If detector coordinates are provided, detector data will also be created
        by converting coronagraph counts to detector pixel scale.

        Args:
            coro_counts (numpy.ndarray):
                Count data for the source in coronagraph pixels.
            source_name (str):
                Name of the source ('img', 'star', 'planet', 'disk').
            ds (xarray.Dataset):
                The dataset to which the source data will be added.
            coro_coords (list of numpy.ndarray):
                List of coordinate arrays for coronagraph data dimensions.
            coro_dims (list of str):
                List of dimension names for coronagraph data.
            det_coords (list of numpy.ndarray or None):
                List of coordinate arrays for detector data dimensions.
            det_dims (list of str or None):
                List of dimension names for detector data.

        Returns:
            xarray.Dataset:
                The updated dataset with the new source data added as DataArrays
                named '{source_name}(coro)' and optionally '{source_name}(det)'.
        """
        # Coronagraph data
        coro_da = xr.DataArray(coro_counts, coords=coro_coords, dims=coro_dims)
        coro_da.name = f"{source_name}(coro)"
        ds = xr.merge([ds, coro_da])

        # Detector data
        if self.scenario.detector_shape is not None:
            counts_det = self.convert_coro_to_detector(coro_counts)
            det_da = xr.DataArray(counts_det, coords=det_coords, dims=det_dims)
            det_da.name = f"{source_name}(det)"
            ds = xr.merge([ds, det_da])

        return ds

    def get_coro_image_shape(self):
        """Get the shape of the coronagraph count/image pixel array.

        The shape depends on the observation settings and includes dimensions for:
        - Time frames (from nframes)
        - Spectral wavelengths (1 if no wavelength dependence, otherwise the
          number of wavelength grid points)
        - Spatial dimensions (coronagraph npixels x npixels)

        The resulting shape is always 4-dimensional: (nframes, nwavelengths,
        npixels, npixels).

        Returns:
            tuple of int:
                The shape of the coronagraph count/image pixel array in the format
                (nframes, nwavelengths, npixels, npixels).
        """
        coro_image_shape = [self.nframes]
        if self.settings.any_wavelength_dependence:
            coro_image_shape.append(len(self.spectral_wavelength_grid))
        else:
            coro_image_shape.append(1)

        coro_image_shape.extend([self.coronagraph.npixels, self.coronagraph.npixels])
        return tuple(coro_image_shape)

    def calc_frame_info(self):
        """Calculate the number of frames and the start times for each frame.

        This method determines how to split the total exposure time into individual
        frames based on the observation settings. If settings.return_frames is True,
        the exposure is divided into multiple frames of duration scenario.frame_time.
        If False, the entire exposure is treated as a single frame for computational
        efficiency (valid due to Poisson statistics).

        Note:
            Partial frames are not currently supported - the exposure time must be
            evenly divisible by the frame time.

        Returns:
            tuple:
                A tuple containing:

                - nframes (int):
                    Number of frames in the exposure.
                - frame_start_times (astropy.time.Time):
                    Array of start times for each frame, based on the scenario
                    start time and frame duration.
        """
        if self.settings.return_frames:
            partial_frame, full_frames = np.modf(
                self.scenario.exposure_time_s / self.scenario.frame_time_s
            )
            if partial_frame != 0:
                raise ("Warning! Partial frames are not implemented yet!")
            frame_time = self.scenario.frame_time
        else:
            # If we're not returning frames, we can simulate this as one frame
            # one call due to its Poisson nature
            full_frames = 1
            frame_time = self.scenario.exposure_time
        start_jd = self.scenario.start_time.jd
        frame_d = frame_time.to_value(u.d)
        jd_vals = [start_jd + frame_d * i for i in range(full_frames.astype(int))]
        frame_start_times = Time(jd_vals, format="jd")

        # Setting up the proper shape of the array that counts the photons
        nframes = int(full_frames)
        return nframes, frame_start_times

    def coro_det_coords_and_dims(self):
        """Create the coordinates and dimensions for the coronagraph and detector data.

        Returns:
            coro_coords (list of np.arrays):
                Coronagraph data coordinates.
            coro_dims (list of strs):
                Coronagraph data dimensions.
            det_coords (list of np.arrays):
                Detector data coordinates.
            det_dims (list of strs):
                Detector data dimensions.
        """
        coro_pixel_arr = np.arange(self.coronagraph.npixels)

        if self.settings.any_wavelength_dependence:
            wavelength_coords = self.spectral_wavelength_grid.to(u.nm).value
        else:
            wavelength_coords = [self.scenario.central_wavelength.to(u.nm).value]

        # Coronagraph coordinates and dimensions
        coro_coords = [
            self.frame_start_times.datetime64,
            wavelength_coords,
            coro_pixel_arr,
            coro_pixel_arr,
        ]
        coro_dims = ["time", "spectral_wavelength(nm)", "xpix(coro)", "ypix(coro)"]

        # Detector coordinates and dimensions
        if self.scenario.has_detector:
            detector_xpix_arr = np.arange(self.scenario.detector_shape[0])
            detector_ypix_arr = np.arange(self.scenario.detector_shape[1])
            det_coords = coro_coords[:-2] + [detector_xpix_arr, detector_ypix_arr]
            det_dims = coro_dims[:-2] + ["xpix(det)", "ypix(det)"]
        else:
            det_coords, det_dims = None, None

        return coro_coords, coro_dims, det_coords, det_dims

    def final_coords_and_dims(self):
        """Create the coordinates and dimensions for the final photon dataset.

        Returns:
            final_coords (list of np.arrays):
                Coordinates for the final dataset.
            final_dims (list of strs):
                Dimensions for the final dataset.
        """
        coro_coords, coro_dims, det_coords, det_dims = self.coro_det_coords_and_dims()
        final_coords, final_dims = coro_coords.copy(), coro_dims.copy()

        if not self.settings.return_frames and "time" in coro_dims:
            time_ind = np.argwhere(np.array(final_dims) == "time")[0][0]
            final_dims.remove("time")
            final_coords.pop(time_ind)

        if not self.settings.return_spectrum and "spectral_wavelength(nm)" in coro_dims:
            # Remove the wavelength dimension if we're not returning the spectrum
            # since it will be summed over
            wave_ind = np.argwhere(np.array(final_dims) == "spectral_wavelength(nm)")[
                0
            ][0]
            final_dims.remove("spectral_wavelength(nm)")
            final_coords.pop(wave_ind)

        if self.scenario.has_detector:
            final_coords += det_coords[-2:]
            final_dims += det_dims[-2:]

        return final_coords, final_dims

    def convert_coro_to_detector(self, coro_counts):
        """Convert coronagraph pixel data to the detector pixels.

        Args:
            coro_counts (numpy.ndarray):
                The count/image data in coronagraph pixels (lam/D)

        Returns:
            det_counts (numpy.ndarray):
                The count/image data scaled to the detector pixels (arcsec)
        """
        # Scale the lambda/D pixels to the detector pixels
        if self.settings.any_wavelength_dependence:
            lam = self.spectral_wavelength_grid
        else:
            lam = self.scenario.central_wavelength
        det_counts = util.get_detector_images(
            coro_counts,
            self.coronagraph.pixel_scale,
            lam,
            self.scenario.diameter,
            self.scenario.detector_shape,
            self.scenario.detector_pixel_scale,
        )
        return det_counts

    def plot_count_rates(self):
        """Plot the images at each wavelength and time."""
        min_val = np.min(self.total_count_rate.value * 1e-5)
        max_val = np.max(self.total_count_rate.value)
        norm = LogNorm(vmin=min_val, vmax=max_val)
        for i, time in enumerate(self.times):
            for j, wavelength in enumerate(self.obs_wavelengths):
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                data = [
                    self.total_count_rate[i, j],
                    self.star_count_rate[i, j],
                    self.planet_count_rate[i, j],
                    self.disk_count_rate[i, j],
                ]
                names = ["Total", "Star", "Planet", "Disk"]
                inclusion = [
                    True,
                    self.settings.include_star,
                    self.settings.include_planets,
                    self.settings.include_disk,
                ]
                for ax, data, name, include in zip(
                    axes.flatten(), data, names, inclusion
                ):
                    if include:
                        ax.imshow(data.value, norm=norm)
                        title = f"{name} count rate"
                    else:
                        title = f"{name} not included"

                    ax.set_title(title)
                    if ax.get_subplotspec().is_first_col():
                        ax.set_ylabel("Pixels")
                    if ax.get_subplotspec().is_last_row():
                        ax.set_xlabel("Pixels")

                # add colorbar of the LogNorm to the right of the figure
                fig.subplots_adjust(right=0.8)
                cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm), cax=cax, label="Photons/second"
                )

                fig.suptitle(
                    f"{time.decimalyear:.2f} {wavelength.to(u.nm).value:.0f} nm"
                )
                save_path = Path(
                    self.save_dir,
                    f"{wavelength.to(u.nm).value:.0f}",
                    # , "images"
                )
                if not save_path.exists():
                    save_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    Path(
                        save_path,
                        # f"{wavelength.to(u.nm).value:.0f}_{time.decimalyear:.2f}.png",
                        f"{i:003}.png",
                        bbox_inches="tight",
                    )
                )
                plt.close(fig)

    def snr_check(self, times, noise_factor=0.5):
        """Check the SNR of the images."""
        # Set up count rates for the Poisson noise as a fraction of the max
        noise_count_rate = np.max(self.total_count_rate) * noise_factor
        noise_field = np.ones_like(self.total_count_rate.value) * noise_count_rate

        # Get the brightest pixel
        bright_planet_loc = np.unravel_index(
            np.argmax(self.total_count_rate), self.total_count_rate.shape
        )[::-1]

        # Set up a circular aperture at the bright planet location
        aperture = CircularAperture(bright_planet_loc, r=4)

        # Create an aperture mask
        mask = aperture.to_mask(method="exact")

        # aperture count rate
        aperture_count_rate = np.sum(mask.multiply(self.total_count_rate))
        background_count_rate = np.sum(mask.multiply(noise_field))
        cp = aperture_count_rate
        cb = background_count_rate

        def analytic_snr(t):
            return (cp * t / np.sqrt(cp * t + cb * t)).decompose().value

        snrs = np.zeros_like(times.value)
        analytic_snrs = np.zeros_like(times.value)
        for i, t in enumerate(times):
            self.scenario.exposure_time = t
            self.count_photons()
            expected_photons_per_frame = (
                (noise_field * self.scenario.exposure_time).decompose().value
            )
            noise_frame = np.random.poisson(expected_photons_per_frame)
            combined_image = self.image + noise_frame

            # Get the aperture stats
            aperture_stats = ApertureStats(combined_image, aperture, sigma_clip=None)

            # Create an annulus aperture to compute the mean background level
            annulus = CircularAnnulus(bright_planet_loc, r_in=8, r_out=15)

            # Background statistics
            sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
            bkg_stats = ApertureStats(combined_image, annulus, sigma_clip=sigma_clip)

            # Get the background count in the circular aperture
            bkg_total = bkg_stats.median * aperture.area

            # Subtract the background count from the aperture count
            bkg_subtracted_count = aperture_stats.sum - bkg_total

            snrs[i] = bkg_subtracted_count / np.sqrt(aperture_stats.sum)
            analytic_snrs[i] = analytic_snr(t)

        fig, ax = plt.subplots()
        cmap = mpl.cm.get_cmap("viridis")
        ax.plot(times, analytic_snrs, label="Analytic SNR", color=cmap(0.2))
        ax.scatter(times, snrs, label="Simulated SNR", color=cmap(0.7), zorder=10, s=10)
        ax.set_xlabel("Exposure time (hr)")
        ax.set_ylabel("SNR")
        ax.legend(loc="best")
        ax.set_title("Comparison of simulated and analytic SNR")
        fig.savefig("results/snr_check.png", bbox_inches="tight", dpi=300)
        plt.show()

        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        # max_count = np.max(self.image)
        # axes[0].imshow(
        #     self.image, origin="lower", norm=Normalize(vmin=0, vmax=max_count)
        # )
        # p = axes[1].imshow(
        #     combined_image, origin="lower", norm=Normalize(vmin=0, vmax=max_count)
        # )
        # axes[0].set_title("No noise")
        # axes[1].set_title("With noise and aperture")
        # aperture.plot(color="white", ax=axes[1])
        # annulus.plot(color="red", ax=axes[1])
        # fig.subplots_adjust(right=0.8)
        # cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        # fig.colorbar(p, cax=cax, label="Photon count")
        # fig.savefig("results/snr_check.png", bbox_inches="tight", dpi=300)
        # fig.colorbar(p, ax=axes[1], label="Photon count")

        # bkg_count = bkg_stats.median * aperture.area

        # mean_background = aperstats.mean
        # phot_table = aperture_photometry(combined_image, aperture)

        # aper_stats_bkgsub = ApertureStats(
        #     combined_image, aperture, local_bkg=bkg_stats.median
        # )

        # stats = ApertureStats(combined_image, aperture)


@jax.jit
def compute_disk_image(disk, psfs):
    """JAX compatible function to convolve the disk flux and PSFs.

    Args:
        disk (numpy.ndarray):
            Disk flux density values (Npix, Npix).
        psfs (numpy.ndarray):
            PSF datacube (Npix, Npix, Npix, Npix) where the first two dimensions
            are the x and y offsets of the point source in the PSF.
    """
    return jnp.einsum("ij,ijxy->xy", disk, psfs)


@jax.jit
def compute_disk_image_debug(disk, psfs):
    """Alternative implementation for debugging coordinate issues.

    This version explicitly shows what should happen:
    For each disk pixel (i,j) with flux value disk[i,j],
    add the contribution disk[i,j] * psfs[i,j,:,:] to the final image.
    """
    # Initialize result image
    result = jnp.zeros_like(psfs[0, 0])

    # For each source position in the disk
    npix = disk.shape[0]
    for i in range(npix):
        for j in range(npix):
            # Add the PSF contribution from this source position
            # psfs[i,j,:,:] should be the PSF for a point source at position (i,j)
            result += disk[i, j] * psfs[i, j, :, :]

    return result


@jax.jit
def compute_disk_image_alternative(disk, psfs):
    """Alternative using tensordot - might be more numerically stable."""
    return jnp.tensordot(disk, psfs, axes=([0, 1], [0, 1]))
