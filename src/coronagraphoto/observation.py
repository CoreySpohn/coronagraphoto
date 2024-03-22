import copy
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import xarray as xr
from astropy.stats import SigmaClip
from astropy.time import Time
from exoverses.util import misc
from lod_unit import lod, lod_eq
from matplotlib.colors import LogNorm
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from scipy.ndimage import rotate, shift, zoom
from tqdm import tqdm

from coronagraphoto import util
from coronagraphoto.logger import setup_logger


class Observation:
    def __init__(
        self, coronagraph, system, observing_scenario, settings, logging_level="INFO"
    ):
        """Class to store the parameters of an observation.
        Args:
            coronagraph (Coronagraph object):
                Coronagraph object containing the coronagraph parameters
            system (ExovistaSystem object):
                ExovistaSystem object containing the system parameters
            observing_scenario (ObservingScenario object):
                Object containing the observing scenario parameters
            settings (Settings object):
                Object containing the settings parameters

        """
        self.coronagraph = coronagraph
        self.coronagraph_name = coronagraph.name
        self.system = system
        self.system_name = system.star.name
        self.logger = setup_logger(logging_level)

        self.load_settings(observing_scenario, settings)

        # Create save directory
        self.save_dir = Path("results", system.file.stem, coronagraph.dir.parts[-1])

    def load_settings(self, observing_scenario, settings):
        # self.observing_scenario = observing_scenario
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
            self.logger.info("Creating wavelength grid")
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
            )
        else:
            self.central_transmission = self.scenario.bandpass(
                self.scenario.central_wavelength
            )

        # Solve for illuminated area
        self.illuminated_area = (
            np.pi
            * self.scenario.diameter**2
            / 4.0
            * (1.0 - self.coronagraph.frac_obscured)
        )

    def create_count_rates(self):
        """
        Create the images at the wavelengths and times
        """
        self.logger.info("Creating count rates")

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
            self.logger.info("Creating star count rate")
            self.star_count_rate = self.generic_count_rate_logic(
                self.gen_star_count_rate,
                base_count_rate_arr,
                time_invariant=self.settings.time_invariant_star,
            )
            self.total_count_rate += self.star_count_rate
        else:
            self.logger.info("Not including star")

        if self.settings.include_planets:
            self.logger.info("Creating planets count rate")
            self.planet_count_rate = self.generic_count_rate_logic(
                self.gen_planet_count_rate,
                base_count_rate_arr,
                time_invariant=self.settings.time_invariant_planets,
            )
            self.total_count_rate += self.planet_count_rate
        else:
            self.logger.info("Not including planets")

        if self.settings.include_disk:
            if not self.coronagraph.has_psf_datacube:
                self.coronagraph.get_disk_psfs()
            self.logger.info("Creating disk count rate")
            self.disk_count_rate = self.generic_count_rate_logic(
                self.gen_disk_count_rate,
                base_count_rate_arr,
                time_invariant=self.settings.time_invariant_disk,
            )
            self.total_count_rate += self.disk_count_rate
        else:
            self.logger.info("Not including disk")

    def generic_count_rate_logic(
        self, count_rate_function, object_count_rate, *args, time_invariant=False
    ):
        """
        Streamlined logic to handle input and output array shapes of:
        (nwavelengths, nframes, npixels, npixels), incorporating time invariance.

        Args:
            count_rate_function (callable): Function that computes count rate.
            object_count_rate (numpy.ndarray): Array for computed count rates.
                Shape: (nframes, nwavelengths, npixels, npixels).

        Returns:
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
        """
        Generate the star count rate in photons per second, WITHOUT
        any transmission effects.
        """
        # Compute star count rate in lambda/D
        stellar_diam_lod = self.system.star.angular_diameter.to(
            lod, lod_eq(wavelength, self.scenario.diameter)
        )

        # Get the intensity map I(x,y) at the stellar diameters
        stellar_intens = self.coronagraph.stellar_intens_interp(stellar_diam_lod).T

        # Calculate the star flux density
        star_flux_density = self.system.star.spec_flux_density(wavelength, time).to(
            u.photon / (u.m**2 * u.s * u.nm),
            equivalencies=u.spectral_density(wavelength),
        )

        # Multiply by the count rate term (A*dLambda*T)
        flux_term = (star_flux_density * self.illuminated_area * bandwidth).decompose()

        # Compute star count rate in each pixel
        count_rate = np.multiply(stellar_intens, flux_term).T
        return count_rate

    def gen_planet_count_rate(self, wavelength, time, bandwidth):
        """
        Add planets to the system.
        """
        # Compute planet separations and position angles.
        prop_kwargs = {
            "prop": "nbody",
            "ref_frame": "helio-sky",
        }
        orbit_dataset = self.system.propagate(time, **prop_kwargs)
        xystar = np.array([self.coronagraph.npixels / 2] * 2) * u.pix
        pixscale = (self.coronagraph.pixel_scale * u.pix).to(
            u.arcsec, lod_eq(wavelength, self.scenario.diameter)
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

        # Compute planet flux.
        coro_type = self.coronagraph.type

        planet_flux_density = np.zeros(len(self.system.planets)) * u.Jy
        for i, planet in enumerate(self.system.planets):
            planet_flux_density[i] = planet.spec_flux_density(wavelength, time)

        planet_photon_flux = (
            planet_flux_density.to(
                u.photon / (u.m**2 * u.s * u.nm),
                equivalencies=u.spectral_density(wavelength),
            )
            * self.illuminated_area
            * bandwidth
        ).decompose()

        # for i, planet in enumerate(tqdm(self.system.planets, desc="Adding planets")):
        if coro_type == "1d":
            planet_lod_alphas = planet_alphas.to(
                lod, lod_eq(wavelength, self.scenario.diameter)
            )

            # The planet psfs at each pixel
            planet_psfs = self.coronagraph.offax_psf_interp(planet_lod_alphas)
            rotated_psfs = np.zeros_like(planet_psfs)

            # interpolate in log-space to avoid negative values
            for i, _angle in enumerate(planet_angles):
                rotated_psfs[i] = np.exp(
                    rotate(
                        np.log(planet_psfs[i]),
                        -_angle.to(u.deg).value,
                        reshape=False,
                        mode="nearest",
                        order=5,
                    )
                )

            planet_count_rate = (rotated_psfs.T @ planet_photon_flux).T
        # elif coro_type == "1dno0":
        #     #TODO NOT IMPLEMENTED YET
        #     planet_lod_alphas = np.stack(
        #         [
        #             planet_alphas[i, :].to(lod, lod_eq(wave, self.scenario.diameter))
        #             for wave in self.obs_wavelengths
        #         ]
        #     )
        #     temp = np.sqrt(
        #         planet_lod_alphas**2 - self.coronagraph.offax_psf_offset_y[0] ** 2
        #     )
        #     seps = (
        #         np.dot(plan_seps[i][:, None] * mas2rad, wave_inv[None]) * self.diam
        #     )  # lambda/D
        #     angs
        #     = np.arcsin(self.coronagraph.offax_psf_offset_y[0] / planet_lod_alphas)
        #     rotated_psfs = self.coronagraph.offax_psf_interp(temp)
        #     for j in range(self.scene.Ntime):
        #         for k in range(self.scene.Nwave):
        #             temp[j, k] = np.exp(
        #                 rotate(
        #                     np.log(temp[j, k]),
        #                     plan_angs[i, j] - 90.0 * u.deg + angs[j, k],
        #                     axes=(0, 1),
        #                     reshape=False,
        #                     mode="nearest",
        #                     order=5,
        #                 )
        #             )  # interpolate in log-space to avoid negative values
        #         temp[j] = np.multiply(
        #             temp[j].T, self.scene.fplanet[i, j] * self.oneJ_count_rate
        #         ).T  # ph/s
        # elif coro_type == "2dq":
        #     # TODO NOT IMPLEMENTED YET
        #     temp = (
        #         np.dot(plan_offs[i][:, :, None] * mas2rad, wave_inv[None]) * self.diam
        #     )  # lambda/D
        #     temp = np.swapaxes(temp, 1, 2)  # lambda/D
        #     temp = temp[:, :, ::-1]  # lambda/D
        #     offs = temp.copy()  # lambda/D
        #     # temp = np.exp(self.ln_offax_psf_interp(np.abs(temp)))
        #     temp = self.offax_psf_interp(np.abs(temp))
        #     mask = offs[:, :, 0] < 0.0
        #     temp[mask] = temp[mask, ::-1, :]
        #     mask = offs[:, :, 1] < 0.0
        #     temp[mask] = temp[mask, :, ::-1]
        #     for j in range(self.scene.Ntime):
        #         temp[j] = np.multiply(
        #             temp[j].T, self.scene.fplanet[i, j] * self.oneJ_count_rate
        #         ).T  # ph/s
        # else:
        #     # TODO NOT IMPLEMENTED YET
        #     temp = (
        #         np.dot(plan_offs[i][:, :, None] * mas2rad, wave_inv[None]) * self.diam
        #     )  # lambda/D
        #     temp = np.swapaxes(temp, 1, 2)  # lambda/D
        #     temp = temp[:, :, ::-1]  # lambda/D
        #     # temp = np.exp(self.ln_offax_psf_interp(temp))
        #     temp = self.offax_psf_interp(temp)
        #     for j in range(self.scene.Ntime):
        #         temp[j] = np.multiply(
        #             temp[j].T, self.scene.fplanet[i, j] * self.oneJ_count_rate
        #         ).T  # ph/s
        return planet_count_rate
        # self.planet_count_rate += planet_image
        # self.total_count_rate += self.planet_count_rate

    def gen_disk_count_rate(self, wavelength, time, bandwidth):
        disk_image = self.system.disk.spec_flux_density(wavelength, time)

        # This is the factor to scale the disk image, from exovista, to the
        # coronagraph model size since they do not necessarily have the same
        # pixel scale
        zoom_factor = (
            (1 * u.pixel * self.system.star.pixel_scale.to(u.rad / u.pixel)).to(
                lod, lod_eq(wavelength, self.scenario.diameter)
            )
            / self.coronagraph.pixel_scale
        ).value
        # This is the photons per second
        disk_image_photons = (
            disk_image.to(
                u.photon / (u.m**2 * u.s * u.nm),
                equivalencies=u.spectral_density(wavelength),
            )
            * self.illuminated_area
            * bandwidth
        ).value
        scaled_disk = np.exp(
            zoom(
                np.log(disk_image_photons),
                zoom_factor,
                mode="nearest",
                order=5,
            )
        )

        # Center disk so that (img_pixels-1)/2 is center.
        disk_pixels_is_even = scaled_disk.shape[0] % 2 == 0
        coro_pixels_is_even = self.coronagraph.npixels % 2 == 0
        if disk_pixels_is_even != coro_pixels_is_even:
            scaled_disk = np.pad(scaled_disk, ((0, 1), (0, 1)), mode="edge")
            # interpolate in log-space to avoid negative values
            scaled_disk = np.exp(shift(np.log(scaled_disk), (0.5, 0.5), order=5))
            scaled_disk = scaled_disk[1:-1, 1:-1]

        # Crop disk to coronagraph model size.
        if scaled_disk.shape[0] > self.coronagraph.npixels:
            nn = (scaled_disk.shape[0] - self.coronagraph.npixels) // 2
            scaled_disk = scaled_disk[nn:-nn, nn:-nn]
        else:
            nn = (self.coronagraph.npixels - scaled_disk.shape[0]) // 2
            scaled_disk = np.pad(scaled_disk, ((nn, nn), (nn, nn)), mode="edge")

        # count_rate = np.einsum("ij,ijkl->kl", scaled_disk, psfs) * u.ph / u.s
        scaled_disk = np.ascontiguousarray(scaled_disk)
        count_rate = (
            nb_gen_disk_count_rate(
                scaled_disk, self.coronagraph.psf_datacube, scaled_disk.shape[0]
            )
            * u.ph
            / u.s
        )
        return count_rate

    def count_photons(self):
        """
        Split the exposure time into individual frames, then simulate the
        collection of photons as a Poisson process
        """
        self.logger.info("Creating images")

        coro_image_shape = self.get_coro_image_shape()

        # Create the arrays to store the frame counts for each source
        # if we're returning them
        if self.settings.return_sources:
            frame_counts = np.zeros(coro_image_shape)
            if self.settings.include_star:
                expected_star_photons_per_frame = (
                    (self.star_count_rate * self.scenario.frame_time).decompose().value
                )
                star_frame_counts = np.random.poisson(expected_star_photons_per_frame)
                frame_counts += star_frame_counts
            if self.settings.include_planets:
                expected_planet_photons_per_frame = (
                    (self.planet_count_rate * self.scenario.frame_time)
                    .decompose()
                    .value
                )
                planet_frame_counts = np.random.poisson(
                    expected_planet_photons_per_frame
                )
                frame_counts += planet_frame_counts
            if self.settings.include_disk:
                expected_disk_photons_per_frame = (
                    (self.disk_count_rate * self.scenario.frame_time).decompose().value
                )
                disk_frame_counts = np.random.poisson(expected_disk_photons_per_frame)
                frame_counts += disk_frame_counts
        else:
            # Calculate the expected number of photons per frame
            expected_photons_per_frame = (
                (self.total_count_rate * self.scenario.frame_time).decompose().value
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
        """
        Create and add source DataArrays for both coronagraph and detector to
        the dataset.

        Args:
            ds (xarray.Dataset):
                The dataset to which the source data will be added.
            frame_counts (numpy.ndarray):
                Frame counts for the source.
            source_name (str):
                Name of the source ('img', 'star', 'planet', 'disk').


        Returns:
            xarray.Dataset:
                The updated dataset with the new source data.
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
        """
        Get the shape of the coronagraph count/image pixel array. Depends
        on whether we're calculating the wavelength dependence or not.

        Returns:
            coro_image_shape (tuple of ints):
                The shape of the coronagraph count/image pixel array
        """
        coro_image_shape = [self.nframes]
        if self.settings.any_wavelength_dependence:
            coro_image_shape.append(len(self.spectral_wavelength_grid))
        else:
            coro_image_shape.append(1)

        coro_image_shape.extend([self.coronagraph.npixels, self.coronagraph.npixels])
        return tuple(coro_image_shape)

    def calc_frame_info(self):
        """
        Calculate the number of frames and the length of each individual frame
        in the exposure

        Returns:
            nframes (int):
                Number of frames
            frame_start_times (astropy Time array):
                The start times of each frame
        """
        if self.settings.return_frames:
            partial_frame, full_frames = np.modf(
                (self.scenario.exposure_time / self.scenario.frame_time)
                .decompose()
                .value
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
        frame_d = frame_time.to(u.d).value
        jd_vals = [start_jd + frame_d * i for i in range(full_frames.astype(int))]
        frame_start_times = Time(jd_vals, format="jd")

        # Setting up the proper shape of the array that counts the photons
        nframes = int(full_frames)
        return nframes, frame_start_times

    def coro_det_coords_and_dims(self):
        """
        Create the coordinates and dimensions for the coronagraph and detector
        data.

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
        """
        Create the coordinates and dimensions for the final photon dataset.

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
        """
        Convert coronagraph pixel data to the detector pixels

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
        """
        Plot the images at each wavelength and time
        """
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
        """
        Check the SNR of the images
        """
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


@nb.njit(fastmath=True, parallel=True, cache=True)
def nb_gen_disk_count_rate(disk, psfs, npix):
    assert disk.shape[0] == npix
    assert disk.shape[1] == npix
    assert psfs.shape[0] == npix
    assert psfs.shape[1] == npix
    assert psfs.shape[2] == npix
    assert psfs.shape[3] == npix
    res = np.zeros((npix, npix), dtype=np.float32)
    for i in nb.prange(npix):
        for j in range(npix):
            # i, j represent the offset of the PSF to be multiplied by the disk
            # flux
            res += np.multiply(disk, psfs[i, j])

    return res
