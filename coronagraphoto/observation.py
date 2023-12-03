from itertools import product
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from lod_unit.lod_unit import lod, lod_eq
from matplotlib.colors import LogNorm
from scipy.ndimage import rotate, shift, zoom
from tqdm import tqdm


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
        self.observing_scenario = observing_scenario

        # Load observing scenario parameters
        self.diameter = self.observing_scenario.scenario["diameter"]
        self.obs_wavelength = self.observing_scenario.scenario["wavelength"]
        self.obs_time = self.observing_scenario.scenario["time"]
        self.exposure_time = self.observing_scenario.scenario["exposure_time"]
        self.frame_time = self.observing_scenario.scenario["frame_time"]
        # self.frame_resolution = self.observing_scenario["frame_resolution"]

        self.include_star = self.observing_scenario.scenario.get("include_star")
        self.include_planets = self.observing_scenario.scenario.get("include_planets")
        self.include_disk = self.observing_scenario.scenario.get("include_disk")
        self.bandpass = self.observing_scenario.scenario.get("bandpass")
        self.spectral_resolution = self.observing_scenario.scenario.get(
            "spectral_resolution"
        )

        # Create save directory
        self.save_dir = Path("results", system.file.stem, coronagraph.dir.parts[-1])

        # Create the images
        self.create_count_rate_factor()
        self.create_count_rates()
        # self.plot_count_rates()
        # if self.include_photon_noise:
        self.count_photons()

    def create_count_rate_factor(self):
        self.throughput = self.coronagraph.inst_thruput
        self.illuminated_area = (
            np.pi * self.diameter**2 / 4.0 * (1.0 - self.coronagraph.frac_obscured)
        )
        self.bandwidth = self.coronagraph.frac_bandwidth * self.obs_wavelength
        self.bandwidth_transmission_term = (
            self.illuminated_area * self.bandwidth * self.throughput
        )
        self.count_rate_term = self.illuminated_area * self.bandwidth * self.throughput

    def create_count_rates(self):
        """
        Create the images at the wavelengths and times
        """

        self.count_rates = (
            np.zeros(
                (
                    self.coronagraph.npixels,
                    self.coronagraph.npixels,
                )
            )
            * u.ph
            / u.s
        )

        self.star_count_rate = np.zeros_like(self.count_rates.value) * u.ph / u.s
        self.planet_count_rate = np.zeros_like(self.count_rates.value) * u.ph / u.s
        self.disk_count_rate = np.zeros_like(self.count_rates.value) * u.ph / u.s
        if self.include_star:
            self.add_star_count_rate()
        if self.include_planets:
            self.add_planets_count_rate()
        if self.include_disk:
            self.add_disk_count_rate()

    def add_star_count_rate(self):
        """
        Add a star to the system.
        """
        # Compute star count rate in lambda/D
        stellar_diam_lod = self.system.star.angular_diameter.to(
            lod, lod_eq(self.obs_wavelength, self.diameter)
        )

        # Get the intensity map I(x,y) at the stellar diameters
        stellar_intens = self.coronagraph.stellar_intens_interp(stellar_diam_lod).T

        # Calculate the star flux density
        star_flux_density = self.system.star.spec_flux_density(
            self.obs_wavelength, self.obs_time
        ).to(
            u.photon / (u.m**2 * u.s * u.nm),
            equivalencies=u.spectral_density(self.obs_wavelength),
        )

        # Multiply by the count rate term (A*dLambda*T)
        flux_term = (star_flux_density * self.count_rate_term).decompose()

        # Compute star count rate in each pixel
        self.star_count_rate = np.multiply(stellar_intens, flux_term).T
        self.count_rates += self.star_count_rate

    def add_planets_count_rate(self):
        """
        Add planets to the system.
        """
        # Compute planet separations and position angles.
        xyplanet = np.zeros((len(self.system.planets), 2)) * u.pixel
        for i, planet in enumerate(self.system.planets):
            planet_x = planet._x_pix_interp(self.obs_time) * u.pixel
            planet_y = planet._y_pix_interp(self.obs_time) * u.pixel
            xyplanet[i, 0] = planet_x
            xyplanet[i, 1] = planet_y

        star_x = self.system.star._x_pix_interp(self.obs_time)
        star_y = self.system.star._y_pix_interp(self.obs_time)
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
            planet_flux_density[i] += planet.spec_flux_density(
                self.obs_wavelength, self.obs_time
            )[0]
        planet_photon_flux = (
            planet_flux_density.to(
                u.photon / (u.m**2 * u.s * u.nm),
                equivalencies=u.spectral_density(self.obs_wavelength),
            )
            * self.count_rate_term
        ).decompose()

        # for i, planet in enumerate(tqdm(self.system.planets, desc="Adding planets")):
        if coro_type == "1d":
            planet_lod_alphas = planet_alphas.to(
                lod, lod_eq(self.obs_wavelength, self.diameter)
            )

            # The planet psfs at each pixel
            planet_psfs = self.coronagraph.offax_psf_interp(planet_lod_alphas)
            rotated_psfs = np.zeros_like(planet_psfs)
            # interpolate in log-space to avoid negative values
            for i, _angle in enumerate(planet_angles):
                rotated_psfs[i] = np.exp(
                    rotate(
                        np.log(planet_psfs[i]),
                        _angle,
                        reshape=False,
                        mode="nearest",
                        order=5,
                    )
                )
            planet_image = (rotated_psfs.T @ planet_photon_flux).T
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
        self.planet_count_rate += planet_image
        self.count_rates += self.planet_count_rate

    def add_disk_count_rate(self):
        # Load data cube of spatially dependent PSFs.
        disk_dir = Path(".cache/disks/")
        if not disk_dir.exists():
            disk_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            disk_dir,
            self.coronagraph.dir.name + ".npy",
        )
        if path.exists():
            psfs = np.load(path, allow_pickle=True)
            print("Loaded data cube of spatially dependent PSFs")

        # Compute data cube of spatially dependent PSFs.
        else:
            # Compute pixel grid.
            # lambda/D
            # ramp = ( np.arange(self.coronagraph.npixels) -
            # ((self.coronagraph.npixels - 1) // 2)) * self.pixel_scale
            pixel_lod = (
                (
                    np.arange(self.coronagraph.npixels)
                    - ((self.coronagraph.npixels - 1) // 2)
                )
                * u.pixel
                * self.coronagraph.pixel_scale
            )

            # lambda/D
            # xx, yy = np.meshgrid(ramp, ramp)
            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod)

            # lambda/D
            # rr = np.sqrt(xx**2 + yy**2)
            pixel_dist_lod = np.sqrt(x_lod**2 + y_lod**2)

            # deg
            # tt = np.rad2deg(np.arctan2(xx, yy))
            pixel_angle = np.arctan2(y_lod, x_lod)

            # Compute pixel grid contrast.
            print("   Computing data cube of spatially dependent PSFs")
            # psfs = np.zeros(
            #     (rr.shape[0], rr.shape[1], self.img_pixels, self.img_pixels)
            # )
            psfs = np.zeros(
                (
                    pixel_dist_lod.shape[0],
                    pixel_dist_lod.shape[1],
                    self.coronagraph.npixels,
                    self.coronagraph.npixels,
                )
            )
            # Npsfs = np.prod(rr.shape)
            npsfs = np.prod(pixel_dist_lod.shape)

            pbar = tqdm(total=npsfs, desc="Computing datacube of PSFs at every pixel")

            radially_symmetric_psf = "1d" in self.coronagraph.type
            # Get the PSF (npixel, npixel) of a source at every pixel
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
                        psf = self.coronagraph.ln_offax_psf_interp(psf_eval_dists)
                        temp = np.exp(
                            rotate(
                                psf,
                                rotate_angle,
                                reshape=False,
                                mode="nearest",
                                order=5,
                            )
                        )
                    psfs[i, j] = temp
                    pbar.update(1)

            # Save data cube of spatially dependent PSFs.
            np.save(path, psfs, allow_pickle=True)

        disk_image = self.system.disk.spec_flux_density(
            self.obs_wavelength, self.obs_time
        )
        disk_image_jy = disk_image.to(u.Jy).value

        # Rotate disk so that North is in the direction of the position angle.
        # if self.coronagraph.position_angle != 0.0 * u.deg:
        #     # interpolate in log-space to avoid negative values
        #     disk_image = (
        #         np.exp(
        #             rotate(
        #                 np.log(disk_image_jy),
        #                 self.position_angle,
        #                 axes=(3, 2),
        #                 reshape=False,
        #                 mode="nearest",
        #                 order=5,
        #             )
        #         )
        #         * u.Jy
        #     )
        #     disk_image_jy = disk_image.to(u.Jy).value

        # This is the factor to scale the disk image, from exovista, to the
        # coronagraph model size since they do not necessarily have the same
        # pixel scale
        zoom_factor = (
            (1 * u.pixel * self.system.star.pixel_scale.to(u.rad / u.pixel)).to(
                lod, lod_eq(self.obs_wavelength, self.diameter)
            )
            / self.coronagraph.pixel_scale
        ).value
        # pbar = tqdm(
        #     total=self.ntimes * self.nwavelengths,
        #     desc="Convolving PSF datacube with the disk to create images of the disk",
        # )
        # for j, wavelength in enumerate(self.obs_wavelengths):
        # This is the photons per second
        disk_image_photons = (
            disk_image.to(
                u.photon / (u.m**2 * u.s * u.nm),
                equivalencies=u.spectral_density(self.obs_wavelength),
            )
            * self.count_rate_term
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

        self.disk_count_rate = np.tensordot(scaled_disk, psfs) * u.ph / u.s
        self.count_rates += self.disk_count_rate

    def count_photons(self):
        """
        Split the exposure time into individual frames, then simulate the
        collection of photons as a Poisson process
        """

        partial_frame, full_frames = np.modf(
            (self.exposure_time / self.frame_time).decompose().value
        )
        if partial_frame != 0:
            raise ("Warning! Partial frames are not implemented yet!")
        nframes = int(full_frames)
        frame_counts = np.zeros(
            (
                nframes,
                self.coronagraph.npixels,
                self.coronagraph.npixels,
            )
        )
        expected_photons_per_frame = (
            (self.count_rates * self.frame_time).decompose().value
        )
        pbar = tqdm(
            total=nframes,
            desc="Adding photon noise",
        )
        for i in range(nframes):
            frame = np.random.poisson(expected_photons_per_frame)
            frame_counts[i] = frame
            pbar.update(1)
        self.image = np.sum(frame_counts, axis=0)
        # return image

    def plot_count_rates(self):
        """
        Plot the images at each wavelength and time
        """
        min_val = np.min(self.count_rates.value * 1e-5)
        max_val = np.max(self.count_rates.value)
        norm = LogNorm(vmin=min_val, vmax=max_val)
        for i, time in enumerate(self.obs_times):
            for j, wavelength in enumerate(self.obs_wavelengths):
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                data = [
                    self.count_rates[i, j],
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
