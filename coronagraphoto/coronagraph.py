from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from lod_unit.lod_unit import lod
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import zoom

from coronagraphoto.logger import logger


class Coronagraph:
    def __init__(self, dir):
        """
        Args:
            dir (str):
                Coronagraph directory. Must have fits files
                    stellar_intens.fits - Stellar intensity map
                        Unitless 3d array of the stellar intensity function I,
                        as a function of (x, y) pixel coordinates and the
                        stellar angular diameter theta_star. Values in the map
                        are equal to the stellar count rate in a given pixel
                        divided by the total stellar count rate entering the
                        coronagraph. Does not include reductions such as QE, as
                        in without a coronagraph the total of I is unity.
                    stellar_intens_diam_list.fits - Stellar diameter list
                        A vector of stellar diameter values (lam/D) corresponding
                        to the theta_star values in stellar_intens.

                    offax_psf_offset_list - The off-axis PSF list
                    offax_psf - PSF of off-axis sources
                    sky_trans - Sky transmission data
                    stellar_intens_1 - on-axis data
                    stellar_intens_2 - reference data
                    stellar_intens_diam_list - stellar intensity diameters
            verbose (Bool):
                Whether to use print statements
        """

        ###################
        # Read input data #
        ###################
        logger.info("Creating coronagraph")

        dir = Path(dir)
        self.name = dir.stem
        # Get header and calculate the lambda/D value
        stellar_intens_header = pyfits.getheader(Path(dir, "stellar_intens_1.fits"), 0)

        # Stellar intensity of the star being observed as function of stellar
        # angular diameter (unitless)
        self.stellar_intens = pyfits.getdata(Path(dir, "stellar_intens_1.fits"), 0)
        # the stellar angular diameters in stellar_intens_1 in units of lambda/D
        self.stellar_intens_diam_list = (
            pyfits.getdata(Path(dir, "stellar_intens_diam_list.fits"), 0) * lod
        )

        # Get pixel scale with units
        self.pixel_scale = stellar_intens_header["PIXSCALE"] * lod / u.pixel

        # Load off-axis data (e.g. the planet) (unitless intensity maps)
        self.offax_psf = pyfits.getdata(Path(dir, "offax_psf.fits"), 0)

        # The offset list here is in units of lambda/D
        self.offax_psf_offset_list = (
            pyfits.getdata(Path(dir, "offax_psf_offset_list.fits"), 0) * lod
        )

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################
        if (self.offax_psf_offset_list.shape[1] != 2) and (
            self.offax_psf_offset_list.shape[0] == 2
        ):
            # This condition occurs when the offax_psf_offset_list is transposed
            # from the expected format for radially symmetric coronagraphs
            self.offax_psf_offset_list = self.offax_psf_offset_list.T

        # Check that we have both x and y offset information (even if there
        # is only one axis with multiple values)
        if self.offax_psf_offset_list.shape[1] != 2:
            raise UserWarning("Array offax_psf_offset_list should have 2 columns")

        # Get the unique values of the offset list so that we can format the
        # data into
        self.offax_psf_offset_x = np.unique(self.offax_psf_offset_list[:, 0])
        self.offax_psf_offset_y = np.unique(self.offax_psf_offset_list[:, 1])

        if (len(self.offax_psf_offset_x) == 1) and (
            self.offax_psf_offset_x[0] == 0 * lod
        ):
            self.type = "1d"
            # Instead of handling angles for 1dy, swap the x and y
            self.offax_psf_offset_x, self.offax_psf_offset_y = (
                self.offax_psf_offset_y,
                self.offax_psf_offset_x,
            )

            # self.offax_psf_base_angle = 90.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif (len(self.offax_psf_offset_y) == 1) and (
            self.offax_psf_offset_y[0] == 0 * lod
        ):
            self.type = "1d"
            # self.offax_psf_base_angle = 0.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif len(self.offax_psf_offset_x) == 1:
            # 1 dimensional with offset (e.g. no offset=0)
            self.type = "1dno0"
            self.offax_psf_offset_x, self.offax_psf_offset_y = (
                self.offax_psf_offset_y,
                self.offax_psf_offset_x,
            )
            # self.offax_psf_base_angle = 90.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif len(self.offax_psf_offset_y) == 1:
            self.type = "1dno0"
            # self.offax_psf_base_angle = 0.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif np.min(self.offax_psf_offset_list) >= 0 * lod:
            self.type = "2dq"
            # self.offax_psf_base_angle = 0.0 * u.deg
            # logger.info(
            #     f"Quarterly symmetric response --> reflecting PSFs ({self.type})"
            # )
            logger.info("Coronagraph is quarterly symmetric")
        else:
            self.type = "2df"
            # self.offax_psf_base_angle = 0.0 * u.deg
            logger.info("Coronagraph response is full 2D")

        ############
        # Clean up #
        ############
        # Set position angle, this uses the offax_psf_base_angle value which is
        # set above

        # Center coronagraph model so that image size is odd and central pixel is center
        # TODO: Automate this process
        verified_coronagraph_models = [
            "LUVOIR-A_APLC_10bw_smallFPM_2021-05-05_Dyn10pm-nostaticabb",
            "LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb",
            "LUVOIR-B-VC6_timeseries",
            "LUVOIR-B_VC6_timeseries",
        ]
        if dir.parts[-1] in verified_coronagraph_models:
            self.stellar_intens = self.stellar_intens[:, 1:, 1:]
            self.offax_psf = self.offax_psf[:, :-1, 1:]
        else:
            raise UserWarning(
                "Please validate centering for this unknown coronagraph model"
            )

        # Simulation parameters
        self.dir = dir

        #########################################################################
        # Interpolate coronagraph model (in log space to avoid negative values) #
        #########################################################################
        # Fill value for interpolation
        fill = np.log(1e-100)

        # interpolate stellar data
        self.ln_stellar_intens_interp = interp1d(
            self.stellar_intens_diam_list,
            np.log(self.stellar_intens),
            kind="cubic",
            axis=0,
            bounds_error=False,
            fill_value=fill,
        )
        self.stellar_intens_interp = lambda stellar_diam: np.exp(
            self.ln_stellar_intens_interp(stellar_diam)
        )

        # interpolate planet data depending on type
        if "1" in self.type:
            # Always set up to interpolate along the x axis
            self.ln_offax_psf_interp = interp1d(
                self.offax_psf_offset_list[:, 0],
                np.log(self.offax_psf),
                kind="cubic",
                axis=0,
                bounds_error=False,
                fill_value=fill,
            )
        else:
            zz_temp = self.offax_psf.reshape(
                self.offax_psf_offset_x.shape[0],
                self.offax_psf_offset_y.shape[0],
                self.offax_psf.shape[1],
                self.offax_psf.shape[2],
            )
            if self.type == "2dq":
                # Reflect PSFs to cover the x = 0 and y = 0 axes.
                offax_psf_offset_x = np.append(
                    -self.offax_psf_offset_x[0], self.offax_psf_offset_x
                )
                offax_psf_offset_y = np.append(
                    -self.offax_psf_offset_y[0], self.offax_psf_offset_y
                )
                zz = np.pad(zz_temp, ((1, 0), (1, 0), (0, 0), (0, 0)))
                zz[0, 1:] = zz_temp[0, :, ::-1, :]
                zz[1:, 0] = zz_temp[:, 0, :, ::-1]
                zz[0, 0] = zz_temp[0, 0, ::-1, ::-1]

                self.ln_offax_psf_interp = RegularGridInterpolator(
                    (offax_psf_offset_x, offax_psf_offset_y),
                    np.log(zz),
                    method="linear",
                    bounds_error=False,
                    fill_value=fill,
                )
            else:
                # This section included references to non-class attributes for
                # offax_psf_offset_x and offax_psf_offset_y. I think it meant
                # to be the class attributes
                self.ln_offax_psf_interp = RegularGridInterpolator(
                    (self.offax_psf_offset_x, self.offax_psf_offset_y),
                    np.log(zz_temp),
                    method="linear",
                    bounds_error=False,
                    fill_value=fill,
                )
        self.offax_psf_interp = lambda coordinate: np.exp(
            self.ln_offax_psf_interp(coordinate)
        )

        ##################################################
        # Get remaining parameters and throughput values #
        ##################################################

        # Gets the number of pixels in the image
        self.img_pixels = self.stellar_intens.shape[1] * u.pixel
        self.npixels = self.img_pixels.value.astype(int)

        # Photometric parameters.
        head = pyfits.getheader(Path(dir, "stellar_intens_1.fits"), 0)

        # fractional obscuration
        self.frac_obscured = head["OBSCURED"]
        # print(f"Fractional obscuration = {self.frac_obscured:.3f}")

        # fractional bandpass
        self.frac_bandwidth = (head["MAXLAM"] - head["MINLAM"]) / head["LAMBDA"]
        # print(f"Fractional bandpass = {self.frac_bandwidth:.3f}")

        # instrument throughput
        # TODO: Why is this here if its hardcoded?
        # self.inst_thruput = 1.0
        # print(f"Instrument throughput = {self.inst_thruput:.3f}")

        # Calculate coronagraph throughput
        # self.coro_thruput = self.get_coro_thruput(plot=False)
        # print(f"Coronagraph throughput = {self.coro_thruput:.3f}")

    def get_coro_thruput(self, aperture_radius_lod=0.8, oversample=100, plot=True):
        """
        Get coronagraph throughput
        Args:
            aperture_radius (float):
                Circular aperture radius, in lambda/D (I think)
            oversample (int):
                Oversampling factor for interpolation
            plot (Boolean):
                Whether to plot the coronagraph throughput
        Returns:
            coro_thruput (float):
                Coronagraph throughput
        """
        # Add units
        aperture_radius = aperture_radius_lod * lod

        # Compute off-axis PSF at the median separation value
        # Previously was labeled half max, but there is no guarantee the
        # separations are equally spaced
        if len(self.offax_psf_offset_x) != 1:
            med_offset = self.offax_psf_offset_x[self.offax_psf_offset_x.shape[0] // 2]
        elif len(self.offax_psf_offset_y) != 1:
            med_offset = self.offax_psf_offset_y[self.offax_psf_offset_y.shape[0] // 2]
        else:
            raise UserWarning(
                (
                    "Array offax_psf_offset_list should have more than 1"
                    " unique element for at least one axis"
                )
            )
        # Create (x, y) coordiantes of the aperture in lam/D
        # if self.type in ["1dx", "1dxo"]:
        aperture_pos = u.Quantity([med_offset, self.offax_psf_offset_y[0]])
        # elif self.type in ["1dy", "1dyo"]:
        #     aperture_pos = u.Quantity([self.offax_psf_offset_x[0], med_offset])

        # Create image
        imgs = self.offax_psf_interp(med_offset)

        # Compute aperture position and radius on subarray in pixels.
        # This was 3 times the aperture radius in pixels, I don't know why 3
        # Npix = int(np.ceil(3 * aperture_radius / self.pixel_scale))
        aperture_radius_pix = np.ceil(
            3 * aperture_radius / self.pixel_scale
        ).value.astype(int)

        aperture_pos_pix = (
            (aperture_pos / self.pixel_scale).value + (imgs.shape[0] - 1) / 2
        ).astype(int)
        subarr = imgs[
            aperture_pos_pix[1]
            - aperture_radius_pix : aperture_pos_pix[1]
            + aperture_radius_pix
            + 1,
            aperture_pos_pix[0]
            - aperture_radius_pix : aperture_pos_pix[0]
            + aperture_radius_pix
            + 1,
        ]
        # (aperture_pos / self.pixel_scale + (imgs.shape[0] - 1) / 2.0)
        # This doesn't make sense to me
        pos_subarr = [0, 0] + aperture_radius_pix
        rad_subarr = aperture_radius / self.pixel_scale

        # Compute aperture position and radius on oversampled subarray in pixels.
        norm = np.sum(subarr)
        subarr_zoom = zoom(subarr, oversample, mode="nearest", order=5)
        subarr_zoom *= norm / np.sum(subarr_zoom)
        pos_subarr_zoom = pos_subarr * oversample + (oversample - 1.0) / 2.0
        rad_subarr_zoom = rad_subarr * oversample

        # Compute aperture on oversampled subarray in pixels.
        ramp = np.arange(subarr_zoom.shape[0])
        offax_psf_offset_x, yy = np.meshgrid(ramp, ramp)
        aptr = (
            np.sqrt(
                (offax_psf_offset_x - pos_subarr_zoom[0]) ** 2
                + (yy - pos_subarr_zoom[1]) ** 2
            )
            <= rad_subarr_zoom.value
        )

        # Compute coronagraph throughput
        coro_thruput = np.sum(subarr_zoom[aptr])

        # Plot
        if plot:
            ext = (self.img_pixels / 2.0 * self.pixel_scale).value
            f, axes = plt.subplots(2, 2, figsize=(4.8 * 2, 4.8 * 2))
            ax = axes.flatten()
            ps = []
            ps.append(ax[0].imshow(imgs, origin="lower", extent=(-ext, ext, -ext, ext)))
            ps.append(ax[1].imshow(subarr, origin="lower"))
            ps.append(ax[2].imshow(subarr_zoom, origin="lower"))
            ps.append(ax[3].imshow(aptr, origin="lower"))
            circle_info = [
                (aperture_pos.value, aperture_radius.value),
                (pos_subarr, rad_subarr.value),
                (pos_subarr_zoom, rad_subarr_zoom.value),
                (pos_subarr_zoom, rad_subarr_zoom.value),
            ]
            coords = ["$\lambda/D$", "pix", "pix (oversampled)", "pix (oversampled)"]
            titles = ["Image", "PSF", "Oversampled PSF", "Oversampled aperture"]
            cbar_labels = [
                "Relative flux",
                "Relative flux",
                "Relative flux",
                "Transmission",
            ]
            for _ax, coord, circle, title, p, cbar_label in zip(
                ax, coords, circle_info, titles, ps, cbar_labels
            ):
                _ax.set_xlabel(f"$\Delta$RA [{coord}]")
                _ax.set_ylabel(f"$\Delta$DEC [{coord}]")
                _a = plt.Circle(circle[0], circle[1], fc="none", ec="red")
                _ax.add_patch(_a)
                _ax.set_title(title)
                _c = f.colorbar(p, ax=_ax, fraction=0.046, pad=0.04)
                _c.set_label(cbar_label, rotation=270, labelpad=20)

            plt.tight_layout()
            plt.show()
            plt.close()

        return coro_thruput
