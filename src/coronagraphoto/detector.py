"""Detector modeling for coronagraph simulations.

This module provides the `Detector` class for simulating the behavior of a
detector in a coronagraphic instrument. It handles the conversion of incident
photons from the coronagraph optics into detected photo-electrons, including
the introduction of various noise sources.

Key features include:
-   **Pixel Scale and Shape**: Defines the physical layout of the detector array.
-   **Quantum Efficiency (QE)**: Models the wavelength-dependent efficiency of
    photon-to-electron conversion.
-   **Noise Simulation**: Includes models for:
    -   Dark current: Thermally generated electrons.
    -   Read noise: Noise introduced during the readout process.
    -   Clock-induced charge (CIC): Spurious charge generated during pixel clocking.
-   **Image Resampling**: Converts images from the coronagraph's native resolution
    (in units of lambda/D) to the detector's pixel grid.

The `BaseDetector` class provides a foundational structure, while the `Detector`
class offers a straightforward implementation. The module uses `astropy.units`
for physical quantities and `numpy` for numerical calculations.
"""

from abc import ABC

import astropy.units as u
import numpy as np
import xarray as xr
from lod_unit import lod_eq
from scipy.ndimage import zoom

as_per_pix = u.arcsec / u.pix


class BaseDetector:
    """Base class for a detector.

    This class defines the interface for all detector models. It handles the
    conversion of incident photons to photo-electrons, the generation of noise,
    and the creation of the final detector image.
    """

    def __init__(
        self,
        pixel_scale,
        shape,
        quantum_efficiency=1.0,
        dark_current_rate=0 * u.electron / u.s,
        read_noise=0 * u.electron,
        cic_rate=0 * u.electron,
    ):
        """Initialize the BaseDetector.

        Args:
            pixel_scale (astropy.units.Quantity):
                The angular scale of a pixel.
            shape (tuple of int):
                The shape of the detector array in pixels (height, width).
            quantum_efficiency (float):
                The fraction of incident photons converted to photo-electrons.
                Value should be between 0 and 1.
            dark_current_rate (astropy.units.Quantity):
                The rate at which dark current electrons are generated per
                pixel, typically in electrons/second.
            read_noise (astropy.units.Quantity):
                The number of electrons generated per pixel per readout.
            cic_rate (astropy.units.Quantity):
                The number of electrons generated per pixel per frame.
        """
        self.pixel_scale = pixel_scale
        self.shape = tuple(int(x) for x in shape)
        self.quantum_efficiency = quantum_efficiency
        self.dark_current_rate = dark_current_rate
        self.read_noise = read_noise
        self.cic_rate = cic_rate

    def apply_qe(self, incident_photons):
        """Apply quantum efficiency to incident photons.

        This method simulates the conversion of photons to photo-electrons by
        performing a binomial draw for each pixel, determined by the detector's
        quantum efficiency.

        Args:
            incident_photons (numpy.ndarray):
                An array representing the number of photons hitting each pixel.

        Returns:
            numpy.ndarray:
                An array representing the number of photo-electrons generated.
        """
        return np.random.binomial(
            n=incident_photons.astype(int), p=self.quantum_efficiency
        ).astype(float)

    def get_dark_current(self, frame_time, shape):
        """Generate dark current counts for a given shape.

        Args:
            frame_time (astropy.units.Quantity):
                The exposure time of a single frame.
            shape (tuple):
                The shape of the output array.

        Returns:
            numpy.ndarray:
                An array of dark current counts.
        """
        # The dark current rate is per pixel. To model it in a datacube,
        # we divide the rate by the number of wavelength channels.
        # The sum of n Poisson variables with mean M/n is a Poisson
        # variable with mean M, so this is statistically correct.
        n_lam = shape[1]
        expected_dark_counts_per_bin = (
            (self.dark_current_rate / n_lam) * frame_time
        ).to_value(u.electron)

        return np.random.poisson(expected_dark_counts_per_bin, shape).astype(float)

    def get_read_noise(self, shape):
        """Generate read noise for a given shape.

        Args:
            shape (tuple):
                The shape of the output array.

        Returns:
            numpy.ndarray:
                An array of read noise values.
        """
        # TODO: Implement read noise
        return np.zeros(shape)

    def get_cic(self, shape):
        """Generate clock-induced charge for a given shape.

        Args:
            shape (tuple):
                The shape of the output array.

        Returns:
            numpy.ndarray:
                An array of clock-induced charge counts.
        """
        # CIC is a Poisson process, so like dark current, we divide the
        # rate by the number of wavelength channels.
        n_lam = shape[1]
        cic_per_bin = self.cic_rate.to_value(u.electron) / n_lam
        return np.random.poisson(cic_per_bin, shape).astype(float)

    def add_detector_effects(
        self,
        dataset,
        scene_rates_coro,
        coronagraph,
        obs_scenario,
        settings,
        det_coords,
        det_dims,
    ):
        """Process count rates and add all detector effects to the dataset.

        Args:
            dataset (xarray.Dataset):
                The observation dataset to which detector data will be added.
            scene_rates_coro (dict):
                A dictionary containing the coronagraph-plane count rate arrays for
                each astrophysical component (e.g., 'star', 'planet', 'disk').
            coronagraph (yippy.Coronagraph):
                A yippy coronagraph object that's being used to generate the images.
            obs_scenario (ObservingScenario):
                The observing scenario, providing timing and configuration.
            settings (Settings):
                The simulation settings.
            det_coords (list):
                A list of coordinate arrays for the detector plane.
            det_dims (list):
                A list of dimension names for the detector plane.

        Returns:
            xarray.Dataset:
                The updated dataset with all detector-plane data added.
        """
        det_image_shape = tuple(int(len(c)) for c in det_coords)

        if settings.any_wavelength_dependence:
            lam = obs_scenario.spectral_wavelength_grid
        else:
            lam = obs_scenario.central_wavelength

        scene_electrons = {}
        for name, rate_coro in scene_rates_coro.items():
            if rate_coro is not None:
                rate_det = self._resample_to_detector(
                    rate_coro.value,
                    coronagraph.pixel_scale,
                    lam,
                    obs_scenario.diameter,
                )
                scene_electrons[name] = self._rate_to_electrons(
                    rate_det, obs_scenario.frame_time_s, det_image_shape
                )
            else:
                scene_electrons[name] = np.zeros(det_image_shape)

        total_astro_electrons = np.sum(list(scene_electrons.values()), axis=0)
        dataset = self._add_det_electrons_to_dataset(
            total_astro_electrons, "scene", dataset, det_coords, det_dims
        )
        if settings.return_sources:
            for name, data in scene_electrons.items():
                dataset = self._add_det_electrons_to_dataset(
                    data, name, dataset, det_coords, det_dims
                )

        # --- Detector Noise Simulation ---
        noise_components = {
            "dark_current": self.get_dark_current(
                obs_scenario.frame_time, det_image_shape
            ),
            "read_noise": self.get_read_noise(det_image_shape),
            "cic": self.get_cic(det_image_shape),
        }
        for name, data in noise_components.items():
            dataset = self._add_det_electrons_to_dataset(
                data, name, dataset, det_coords, det_dims
            )

        # --- Final Detector Image ---
        total_noise_electrons = np.sum(list(noise_components.values()), axis=0)
        total_detector_signal = total_astro_electrons + total_noise_electrons
        dataset = self._add_det_electrons_to_dataset(
            total_detector_signal, "image", dataset, det_coords, det_dims
        )
        return dataset

    def _rate_to_electrons(self, rate_det, frame_time_s, shape):
        """Convert a detector-plane rate to photo-electrons.

        Args:
            rate_det (numpy.ndarray):
                The count rate on the detector plane.
            frame_time_s (astropy.units.Quantity):
                The exposure time of a single frame in seconds.
            shape (tuple):
                The shape of the output array.

        Returns:
            numpy.ndarray:
                An array of photo-electron counts.
        """
        if rate_det is None:
            return np.zeros(shape)

        # Convert rate to incident photons and then to photo-electrons
        inc_photons = np.random.poisson(rate_det * frame_time_s)
        return self.apply_qe(inc_photons)

    def _add_det_electrons_to_dataset(self, data, name, ds, coords, dims):
        """Add a detector-scale electron count DataArray to the dataset.

        Args:
            data (numpy.ndarray):
                The data to add to the dataset.
            name (str):
                The name of the data array.
            ds (xarray.Dataset):
                The dataset to add the data to.
            coords (list):
                A list of coordinate arrays for the detector plane.
            dims (list):
                A list of dimension names for the detector plane.

        Returns:
            xarray.Dataset:
                The updated dataset.
        """
        da = xr.DataArray(data, coords=coords, dims=dims)
        da.attrs["units"] = "electron"
        da.name = f"{name}(det)"
        return xr.merge([ds, da])

    def _resample_to_detector(self, lod_arr, lod_scale, lam, D):
        """Resample images from coronagraph (lambda/D) to detector pixels.

        Args:
            lod_arr (numpy.ndarray):
                The array of images in lambda/D space.
            lod_scale (astropy.units.Quantity):
                The pixel scale of the lambda/D images.
            lam (astropy.units.Quantity):
                The wavelength of the observation.
            D (astropy.units.Quantity):
                The diameter of the telescope.

        Returns:
            numpy.ndarray:
                The array of images resampled to the detector pixel scale.
        """
        nframes = lod_arr.shape[0]
        is_spectral_cube = len(lod_arr.shape) == 4

        if is_spectral_cube:
            nlambda = lod_arr.shape[1]
            shape = (int(nframes), int(nlambda), int(self.shape[0]), int(self.shape[1]))
            final_images = np.zeros(shape)

            if not lam.isscalar and len(lam) != nlambda:
                raise ValueError("Length of lam must match nlambda in lod_arr")

            for frame_idx in range(nframes):
                for lambda_idx in range(nlambda):
                    wavelength = lam if lam.isscalar else lam[lambda_idx]
                    final_images[frame_idx, lambda_idx] = self._resample_single(
                        lod_arr[frame_idx, lambda_idx], lod_scale, wavelength, D
                    )
        else:
            shape = (int(nframes), int(self.shape[0]), int(self.shape[1]))
            final_images = np.zeros(shape)
            for frame_idx in range(nframes):
                final_images[frame_idx] = self._resample_single(
                    lod_arr[frame_idx], lod_scale, lam, D
                )
        return final_images

    def _resample_single(self, image, lod_scale, wavelength, diam):
        """Resample a single image from lambda/D to detector pixels.

        Args:
            image (numpy.ndarray):
                The image in lambda/D space.
            lod_scale (astropy.units.Quantity):
                The pixel scale of the lambda/D image.
            wavelength (astropy.units.Quantity):
                The wavelength of the observation.
            diam (astropy.units.Quantity):
                The diameter of the telescope.

        Returns:
            numpy.ndarray:
                The image resampled to the detector pixel scale.
        """
        lod_scale_in_arcsec = (lod_scale * u.pix).to(
            u.arcsec, lod_eq(wavelength, diam)
        ) / u.pix
        zoom_factor = lod_scale_in_arcsec.to_value(
            as_per_pix
        ) / self.pixel_scale.to_value(as_per_pix)

        scaled_image = np.exp(
            zoom(
                np.log(np.clip(image, 1e-30, None)),
                zoom_factor,
                mode="nearest",
                order=5,
            )
        )
        center_offset = (np.array(scaled_image.shape) - np.array(self.shape)) / 2

        if np.any(center_offset < 0):
            pad_amount = np.abs(np.minimum(center_offset, 0)).astype(int)
            padded_image = np.pad(
                scaled_image,
                ((pad_amount[0], pad_amount[0]), (pad_amount[1], pad_amount[1])),
                mode="constant",
            )
            final_image = (
                padded_image[: self.shape[0], : self.shape[1]]
                if padded_image.shape[0] > self.shape[0]
                else np.pad(
                    padded_image,
                    (
                        (0, self.shape[0] - padded_image.shape[0]),
                        (0, self.shape[1] - padded_image.shape[1]),
                    ),
                    mode="constant",
                )
            )
        else:
            final_image = scaled_image[
                int(center_offset[0]) : int(center_offset[0] + self.shape[0]),
                int(center_offset[1]) : int(center_offset[1] + self.shape[1]),
            ]
        return final_image


class Detector(BaseDetector):
    """A simple detector model that inherits the standard behavior from BaseDetector."""

    pass
