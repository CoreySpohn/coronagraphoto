from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import xarray as xr
from astropy.stats import SigmaClip
from lod_unit.lod_unit import lod, lod_eq
from matplotlib.colors import LogNorm, Normalize
from photutils.aperture import (ApertureStats, CircularAnnulus,
                                CircularAperture, aperture_photometry)
from scipy.ndimage import rotate, shift, zoom
from tqdm import tqdm

from coronagraphoto import util
from coronagraphoto.logger import logger


class Observation:
    def __init__(self, coronagraph, system, observing_scenario):
        """Class to store the parameters of an observation.
        Args:
            coronagraph (Coronagraph object):
                Coronagraph object containing the coronagraph parameters
            system (ExovistaSystem object):
                ExovistaSystem object containing the system parameters
            observing_scenario (dict):
                Dictionary containing the observing scenario parameters

        """
        self.coronagraph = coronagraph
        self.system = system

        self.load_observing_scenario(observing_scenario)

        # Create save directory
        self.save_dir = Path("results", system.file.stem, coronagraph.dir.parts[-1])

        # Flag to indicate whether the disk PSF datacube has been loaded
        self.has_psf_datacube = False

    def load_observing_scenario(self, observing_scenario):
        self.observing_scenario = observing_scenario
        # Load observing scenario parameters
        self.diameter = self.observing_scenario.scenario["diameter"]
        self.obs_wavelength = self.observing_scenario.scenario["wavelength"]
        self.obs_time = self.observing_scenario.scenario["time"]
        self.exposure_time = self.observing_scenario.scenario["exposure_time"]
        self.frame_time = self.observing_scenario.scenario["frame_time"]

        # Load observing scenario flags
        self.include_star = self.observing_scenario.scenario.get("include_star")
        self.include_planets = self.observing_scenario.scenario.get("include_planets")
        self.include_disk = self.observing_scenario.scenario.get("include_disk")
        self.return_frames = self.observing_scenario.scenario.get("return_frames")
        self.return_spectrum = self.observing_scenario.scenario.get("return_spectrum")
        self.return_sources = self.observing_scenario.scenario.get("return_sources")

        # Load observing scenario data
        self.bandpass = self.observing_scenario.scenario.get("bandpass")
        self.spectral_resolution = self.observing_scenario.scenario.get(
            "spectral_resolution"
        )
        self.wavelength_resolved_flux = self.observing_scenario.scenario.get(
            "wavelength_resolved_flux"
        )
        self.wavelength_resolved_transmission = self.observing_scenario.scenario.get(
            "wavelength_resolved_transmission"
        )
        self.any_wavelength_dependence = (
            self.wavelength_resolved_flux or self.wavelength_resolved_transmission
        )
        # Check inputs
        if self.return_spectrum:
            assert (
                self.spectral_resolution is not None
            ), "Must provide a spectral_resolution if return_spectrum is True"
            assert self.any_wavelength_dependence, (
                "One or both of wavelength_resolved_flux and "
                "wavelength_resolved_transmission must be True "
                "if return_spectrum is True"
            )
        # Create the wavelength grid and bandwidth
        if self.any_wavelength_dependence:
            logger.info("Creating wavelength grid")
            (
                self.spectral_wavelength_grid,
                self.bandwidth,
            ) = util.gen_wavelength_grid(self.bandpass, self.spectral_resolution)
        else:
            self.bandwidth = self.bandpass.waveset[-1] - self.bandpass.waveset[0]

        # Create the transmission array (or single value)
        if self.wavelength_resolved_transmission:
            self.transmission = self.bandpass(self.spectral_wavelength_grid)
        else:
            self.transmission = self.bandpass(self.obs_wavelength)

        # Solve for illuminated area
        self.illuminated_area = (
            np.pi * self.diameter**2 / 4.0 * (1.0 - self.coronagraph.frac_obscured)
        )

    def create_count_rates(self):
        """
        Create the images at the wavelengths and times
        """
        logger.info("Creating count rates")

        shape = []
        if self.any_wavelength_dependence:
            shape.append(len(self.spectral_wavelength_grid))
        shape.extend([self.coronagraph.npixels, self.coronagraph.npixels])

        self.total_count_rate = np.zeros(tuple(shape)) * u.ph / u.s

        base_count_rate_arr = np.zeros_like(self.total_count_rate.value) * u.ph / u.s
        if self.include_star:
            logger.info("Creating star count rate")
            self.star_count_rate = self.generic_count_rate_logic(
                self.gen_star_count_rate, base_count_rate_arr
            )
            self.total_count_rate += self.star_count_rate
        else:
            logger.info("Not including star")

        if self.include_planets:
            logger.info("Creating planets count rate")
            self.planet_count_rate = self.generic_count_rate_logic(
                self.gen_planet_count_rate, base_count_rate_arr
            )
            self.total_count_rate += self.planet_count_rate
        else:
            logger.info("Not including planets")

        if self.include_disk:
            if not self.has_psf_datacube:
                self.psf_datacube = self.get_disk_psfs()
            logger.info("Creating disk count rate")
            self.disk_count_rate = self.generic_count_rate_logic(
                self.gen_disk_count_rate, base_count_rate_arr, self.psf_datacube
            )
            self.total_count_rate += self.disk_count_rate
        else:
            logger.info("Not including disk")

    def generic_count_rate_logic(self, count_rate_function, object_count_rate, *args):
        """
        Computes the count rate for an object over a spectral wavelength grid.

        This function calculates the count rate of an object by applying a specified
        count rate function over a spectral wavelength grid. The count rate is computed
        differently based on whether the object has wavelength dependence and whether
        the flux and transmission are wavelength resolved.

        Args:
            count_rate_function (callable): A function that takes a wavelength and
                observation time as inputs and returns the count rate.
            object_count_rate (list or numpy.ndarray): An array to store the computed
                count rates for each wavelength in the spectral wavelength grid.

        Behavior:
            - If the object has wavelength dependence and both flux and transmission
              are wavelength resolved, the count rate is calculated for each wavelength.
            - If only transmission is wavelength resolved, a central count rate
              is calculated and all wavelength dependence is from the varying
              transmission of the bandpass.
            - If there is no wavelength dependence, the count rate is calculated using a
              single wavelength and observation time.

        Returns:
            object_count_rate:
                Accumulates the calculated count rates to the object's total count rate.

        """
        if self.any_wavelength_dependence:
            # If we are calculating the count rate for each wavelength
            if self.wavelength_resolved_flux:
                # This indicates that we're generating the count rate
                # with the flux varying over the wavelength grid
                for i, (_wavelength, _bandwidth) in enumerate(
                    tqdm(
                        zip(self.spectral_wavelength_grid, self.bandwidth),
                        desc="Generating count rate for each wavelength",
                        total=len(self.spectral_wavelength_grid),
                        delay=0.5,
                    )
                ):
                    object_count_rate[i] = count_rate_function(
                        _wavelength, self.obs_time, _bandwidth, *args
                    )
            else:
                # In this case, where we're only using the wavelength resolved
                # transmission, we calculate and then copy the central
                # count rates
                central_bandwidth = self.bandwidth[len(self.bandwidth) // 2]
                central_count_rate = count_rate_function(
                    self.obs_wavelength, self.obs_time, central_bandwidth, *args
                )
                for i, _ in enumerate(self.spectral_wavelength_grid):
                    object_count_rate[i] = central_count_rate

            object_count_rate = np.multiply(self.transmission, object_count_rate.T).T
        else:
            object_count_rate = count_rate_function(
                self.obs_wavelength, self.obs_time, self.bandwidth, *args
            )
            object_count_rate *= self.transmission
        return object_count_rate

    def gen_star_count_rate(self, wavelength, time, bandwidth):
        """
        Generate the star count rate in photons per second, WITHOUT
        any transmission effects.
        """
        # Compute star count rate in lambda/D
        stellar_diam_lod = self.system.star.angular_diameter.to(
            lod, lod_eq(wavelength, self.diameter)
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
        xyplanet = np.zeros((len(self.system.planets), 2)) * u.pixel
        for i, planet in enumerate(self.system.planets):
            planet_x = planet._x_pix_interp(time) * u.pixel
            planet_y = planet._y_pix_interp(time) * u.pixel
            xyplanet[i, 0] = planet_x
            xyplanet[i, 1] = planet_y

        star_x = self.system.star._x_pix_interp(time)
        star_y = self.system.star._y_pix_interp(time)
        xystar = np.array([star_x, star_y]) * u.pixel

        # plan_offs
        planet_xy_separations = (xyplanet - xystar) * self.system.star.pixel_scale
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
            planet_lod_alphas = planet_alphas.to(lod, lod_eq(wavelength, self.diameter))

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
        elif coro_type == "1dno0":
            # TODO NOT IMPLEMENTED YET
            planet_lod_alphas = np.stack(
                [
                    planet_alphas[i, :].to(lod, lod_eq(wave, self.diameter))
                    for wave in self.obs_wavelengths
                ]
            )
            temp = np.sqrt(
                planet_lod_alphas**2 - self.coronagraph.offax_psf_offset_y[0] ** 2
            )
            seps = (
                np.dot(plan_seps[i][:, None] * mas2rad, wave_inv[None]) * self.diam
            )  # lambda/D
            angs = np.arcsin(self.coronagraph.offax_psf_offset_y[0] / planet_lod_alphas)
            rotated_psfs = self.coronagraph.offax_psf_interp(temp)
            for j in range(self.scene.Ntime):
                for k in range(self.scene.Nwave):
                    temp[j, k] = np.exp(
                        rotate(
                            np.log(temp[j, k]),
                            plan_angs[i, j] - 90.0 * u.deg + angs[j, k],
                            axes=(0, 1),
                            reshape=False,
                            mode="nearest",
                            order=5,
                        )
                    )  # interpolate in log-space to avoid negative values
                temp[j] = np.multiply(
                    temp[j].T, self.scene.fplanet[i, j] * self.oneJ_count_rate
                ).T  # ph/s
        elif coro_type == "2dq":
            # TODO NOT IMPLEMENTED YET
            temp = (
                np.dot(plan_offs[i][:, :, None] * mas2rad, wave_inv[None]) * self.diam
            )  # lambda/D
            temp = np.swapaxes(temp, 1, 2)  # lambda/D
            temp = temp[:, :, ::-1]  # lambda/D
            offs = temp.copy()  # lambda/D
            # temp = np.exp(self.ln_offax_psf_interp(np.abs(temp)))
            temp = self.offax_psf_interp(np.abs(temp))
            mask = offs[:, :, 0] < 0.0
            temp[mask] = temp[mask, ::-1, :]
            mask = offs[:, :, 1] < 0.0
            temp[mask] = temp[mask, :, ::-1]
            for j in range(self.scene.Ntime):
                temp[j] = np.multiply(
                    temp[j].T, self.scene.fplanet[i, j] * self.oneJ_count_rate
                ).T  # ph/s
        else:
            # TODO NOT IMPLEMENTED YET
            temp = (
                np.dot(plan_offs[i][:, :, None] * mas2rad, wave_inv[None]) * self.diam
            )  # lambda/D
            temp = np.swapaxes(temp, 1, 2)  # lambda/D
            temp = temp[:, :, ::-1]  # lambda/D
            # temp = np.exp(self.ln_offax_psf_interp(temp))
            temp = self.offax_psf_interp(temp)
            for j in range(self.scene.Ntime):
                temp[j] = np.multiply(
                    temp[j].T, self.scene.fplanet[i, j] * self.oneJ_count_rate
                ).T  # ph/s
        return planet_count_rate
        # self.planet_count_rate += planet_image
        # self.total_count_rate += self.planet_count_rate

    def get_disk_psfs(self):
        """
        Load the disk image from a file or generate it if it doesn't exist
        """
        # Load data cube of spatially dependent PSFs.
        disk_dir = Path(".cache/disks/")
        if not disk_dir.exists():
            disk_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            disk_dir,
            self.coronagraph.dir.name + ".nc",
        )

        coords = {
            "x psf offset (pix)": np.arange(self.coronagraph.npixels),
            "y psf offset (pix)": np.arange(self.coronagraph.npixels),
            "x (pix)": np.arange(self.coronagraph.npixels),
            "y (pix)": np.arange(self.coronagraph.npixels),
        }
        dims = ["x psf offset (pix)", "y psf offset (pix)", "x (pix)", "y (pix)"]
        if path.exists():
            logger.info("Loading data cube of spatially dependent PSFs, please hold...")
            psfs_xr = xr.open_dataarray(path)
        else:
            logger.info(
                "Calculating data cube of spatially dependent PSFs, please hold..."
            )
            # Compute pixel grid.
            # lambda/D
            pixel_lod = (
                (
                    np.arange(self.coronagraph.npixels)
                    - ((self.coronagraph.npixels - 1) // 2)
                )
                * u.pixel
                * self.coronagraph.pixel_scale
            )

            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")

            # lambda/D
            pixel_dist_lod = np.sqrt(x_lod**2 + y_lod**2)

            # deg
            pixel_angle = np.arctan2(y_lod, x_lod)

            # Compute pixel grid contrast.
            psfs_shape = (
                pixel_dist_lod.shape[0],
                pixel_dist_lod.shape[1],
                self.coronagraph.npixels,
                self.coronagraph.npixels,
            )
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            npsfs = np.prod(pixel_dist_lod.shape)

            pbar = tqdm(
                total=npsfs, desc="Computing datacube of PSFs at every pixel", delay=0.5
            )

            radially_symmetric_psf = "1d" in self.coronagraph.type
            # Get the PSF (npixel, npixel) of a source at every pixel

            # Note: intention is that i value maps to x offset and j value maps
            # to y offset
            for i in range(pixel_dist_lod.shape[0]):
                for j in range(pixel_dist_lod.shape[1]):
                    # Basic structure here is to get the distance in lambda/D,
                    # determine whether the psf has to be rotated (if the
                    # coronagraph is defined in 1 dimension), evaluate
                    # the offaxis psf at the distance, then rotate the
                    # image
                    if self.coronagraph.type == "1d":
                        psf_eval_dists = pixel_dist_lod[i, j]
                        rotate_angle = pixel_angle[i, j]
                    elif self.coronagraph.type == "1dno0":
                        psf_eval_dists = np.sqrt(
                            pixel_dist_lod[i, j] ** 2
                            - self.coronagraph.offax_psf_offset_x[0] ** 2
                        )
                        rotate_angle = pixel_angle[i, j] + np.arcsin(
                            self.coronagraph.offax_psf_offset_x[0]
                            / pixel_dist_lod[i, j]
                        )
                    elif self.coronagraph.type == "2dq":
                        # lambda/D
                        temp = np.array([y_lod[i, j], x_lod[i, j]])
                        psf = self.coronagraph.offax_psf_interp(np.abs(temp))[0]
                        if y_lod[i, j] < 0.0:
                            # lambda/D
                            psf = psf[::-1, :]
                        if x_lod[i, j] < 0.0:
                            # lambda/D
                            psf = psf[:, ::-1]
                    else:
                        # lambda/D
                        temp = np.array([y_lod[i, j], x_lod[i, j]])
                        psf = self.coronagraph.offax_psf_interp(temp)[0]

                    if radially_symmetric_psf:
                        # if rotate_angle < 0.0:
                        #     rotate_angle += 2 * np.pi * u.rad
                        psf = self.coronagraph.ln_offax_psf_interp(psf_eval_dists)
                        temp = np.exp(
                            rotate(
                                psf,
                                -rotate_angle.to(u.deg).value,
                                reshape=False,
                                mode="nearest",
                                order=5,
                            )
                        )
                    psfs[i, j] = temp
                    # if i == 10 and j == 100:
                    #     # r1 = rotate(
                    #     #     psf,
                    #     #     -135,
                    #     #     reshape=False,
                    #     #     mode="nearest",
                    #     #     order=5,
                    #     # )
                    #     # r2 = rotate(
                    #     #     psf,
                    #     #     -180,
                    #     #     reshape=False,
                    #     #     mode="nearest",
                    #     #     order=5,
                    #     # )
                    #     # r3 = rotate(
                    #     #     psf,
                    #     #     -225,
                    #     #     reshape=False,
                    #     #     mode="nearest",
                    #     #     order=5,
                    #     # )

                    #     fig, ax = plt.subplots(ncols=2)
                    #     # ax[0].imshow(psf, origin="lower")
                    #     # ax[1].imshow(r1, origin="lower")
                    #     # ax[2].imshow(r2, origin="lower")
                    #     # ax[3].imshow(r3, origin="lower")

                    #     ax[0].imshow(psf, origin="lower")
                    #     ax[1].imshow(temp, origin="lower")
                    #     ax[1].scatter(j, i, color="red", marker="s")
                    #     ax[1].set_title(
                    #         f"Rotated by {-rotate_angle.to(u.deg).value:.2f}"
                    #     )
                    #     plt.show()
                    #     breakpoint()
                    pbar.update(1)

            # Save data cube of spatially dependent PSFs.
            psfs_xr = xr.DataArray(
                psfs,
                coords=coords,
                dims=dims,
            )
            psfs_xr.to_netcdf(path)
        self.has_psf_datacube = True
        return np.ascontiguousarray(psfs_xr)

    def gen_disk_count_rate(self, wavelength, time, bandwidth, psfs):
        disk_image = self.system.disk.spec_flux_density(wavelength, time)

        # This is the factor to scale the disk image, from exovista, to the
        # coronagraph model size since they do not necessarily have the same
        # pixel scale
        zoom_factor = (
            (1 * u.pixel * self.system.star.pixel_scale.to(u.rad / u.pixel)).to(
                lod, lod_eq(wavelength, self.diameter)
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
            nb_gen_disk_count_rate(scaled_disk, psfs, scaled_disk.shape[0]) * u.ph / u.s
        )
        return count_rate

    def count_photons(self):
        """
        Split the exposure time into individual frames, then simulate the
        collection of photons as a Poisson process
        """
        logger.info("Creating images")

        if self.return_frames:
            partial_frame, full_frames = np.modf(
                (self.exposure_time / self.frame_time).decompose().value
            )
            if partial_frame != 0:
                raise ("Warning! Partial frames are not implemented yet!")
            frame_time = self.frame_time
        else:
            # If we're not returning frames, we can simulate this as one frame
            # one call due to its Poisson nature
            full_frames = 1
            frame_time = self.exposure_time

        # Setting up the proper shape of the array that counts the photons
        nframes = int(full_frames)
        shape = [nframes]
        if self.any_wavelength_dependence:
            shape.append(len(self.spectral_wavelength_grid))
        shape.extend([self.coronagraph.npixels, self.coronagraph.npixels])
        frame_counts = np.zeros(tuple(shape))

        # Calculate the expected number of photons per frame
        expected_photons_per_frame = (
            (self.total_count_rate * frame_time).decompose().value
        )
        if self.return_sources:
            if self.include_star:
                star_frame_counts = np.zeros(tuple(shape))
                expected_star_photons_per_frame = (
                    (self.star_count_rate * frame_time).decompose().value
                )
            if self.include_planets:
                planet_frame_counts = np.zeros(tuple(shape))
                expected_planet_photons_per_frame = (
                    (self.planet_count_rate * frame_time).decompose().value
                )
            if self.include_disk:
                disk_frame_counts = np.zeros(tuple(shape))
                expected_disk_photons_per_frame = (
                    (self.disk_count_rate * frame_time).decompose().value
                )

        for i in tqdm(range(nframes), desc="Simulating frames", delay=0.5):
            if self.any_wavelength_dependence:
                for j, _ in enumerate(self.spectral_wavelength_grid):
                    if self.return_sources:
                        if self.include_star:
                            star_frame_counts[i, j] = np.random.poisson(
                                expected_star_photons_per_frame[j]
                            )
                            frame_counts[i, j] += star_frame_counts[i, j]
                        if self.include_planets:
                            planet_frame_counts[i, j] = np.random.poisson(
                                expected_planet_photons_per_frame[j]
                            )
                            frame_counts[i, j] += planet_frame_counts[i, j]
                        if self.include_disk:
                            disk_frame_counts[i, j] = np.random.poisson(
                                expected_disk_photons_per_frame[j]
                            )
                            frame_counts[i, j] += disk_frame_counts[i, j]
                    else:
                        frame = np.random.poisson(expected_photons_per_frame[j])
                        frame_counts[i, j] = frame
            else:
                frame = np.random.poisson(expected_photons_per_frame)
                frame_counts[i] = frame
        pixel_arr = np.arange(self.coronagraph.npixels)

        # Create the xarray to store images with the axes labeled
        frame_coords = np.arange(nframes)
        da_coords = [frame_coords]
        ds_coords = {"frame": frame_coords}
        dims = ["frame"]
        if self.any_wavelength_dependence:
            wavelength_coords = self.spectral_wavelength_grid.to(u.nm).value
            da_coords.append(wavelength_coords)
            dims.append("wavelength (nm)")
            ds_coords["wavelength (nm)"] = wavelength_coords
        da_coords.extend([pixel_arr, pixel_arr])
        ds_coords["x (pix)"] = pixel_arr
        ds_coords["y (pix)"] = pixel_arr
        dims.extend(["x (pix)", "y (pix)"])
        total_da = xr.DataArray(frame_counts, coords=da_coords, dims=dims)

        all_das = {"total": total_da}
        if self.return_sources:
            if self.include_star:
                star_da = xr.DataArray(star_frame_counts, coords=da_coords, dims=dims)
                all_das["star"] = star_da
            if self.include_planets:
                planet_da = xr.DataArray(
                    planet_frame_counts, coords=da_coords, dims=dims
                )
                all_das["planet"] = planet_da
            if self.include_disk:
                disk_da = xr.DataArray(disk_frame_counts, coords=da_coords, dims=dims)
                all_das["disk"] = disk_da
        obs_ds = xr.Dataset(all_das, coords=ds_coords)
        if self.any_wavelength_dependence and not self.return_spectrum:
            # Sum over the wavelength axis if we're not returning the spectrum
            # but we did calculate it
            obs_ds = obs_ds.sum(dim="wavelength (nm)")
        if not self.return_frames:
            # Sum over the frame axis if we're not returning the frames
            obs_ds = obs_ds.sum(dim="frame")

        self.data = obs_ds
        # return obs_ds

    def plot_count_rates(self):
        """
        Plot the images at each wavelength and time
        """
        min_val = np.min(self.total_count_rate.value * 1e-5)
        max_val = np.max(self.total_count_rate.value)
        norm = LogNorm(vmin=min_val, vmax=max_val)
        for i, time in enumerate(self.obs_times):
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
                    self.include_star,
                    self.include_planets,
                    self.include_disk,
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
                    f"{wavelength.to(u.nm).value:.0f}"
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
            self.exposure_time = t
            self.count_photons()
            expected_photons_per_frame = (
                (noise_field * self.exposure_time).decompose().value
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
    # tol = 1e-8
    # disk_inds_to_skip = np.where(disk < tol)
    for i in nb.prange(npix):
        for j in range(npix):
            # i, j represent the offset of the PSF to be multiplied by the disk
            # flux
            res += np.multiply(disk, psfs[i, j])

    return res
