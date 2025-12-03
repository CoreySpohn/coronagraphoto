"""Speckle map class."""

from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from scipy.ndimage import median_filter

from coronagraphoto.transforms.image_transforms import flux_conserving_affine

photon_per_sec_unit = u.photon / u.s


class SpeckleMap:
    """A class to generate speckle maps from post-ZWFS images.

    This class loads post-ZWFS images from a .mat file, splits them into two
    halves, and generates speckle maps by subtracting a random image from the
    first half from a random image from the second half.

    Args:
        filepath (Path | str):
            The path to the .mat file containing the post-ZWFS images.
    """

    def __init__(
        self, filepath: Path | str, coronagraph, mult_factor=1, do_subtraction=True
    ):
        """Initialize the SpeckleMap class."""
        with h5py.File(filepath, "r") as data:
            post_zwfs_images = np.array(data["post_zwfs_images"])
            speckle_pixscale = data["xv"][1] - data["xv"][0]
            speckle_shape = post_zwfs_images.shape[1:]

        self.do_subtraction = do_subtraction
        # Data may be stored as (time, y, x), but we want (y, x, time)
        if post_zwfs_images.shape[0] > post_zwfs_images.shape[1]:
            post_zwfs_images = post_zwfs_images.transpose(1, 2, 0)

        # Remove frames 403-436 because of the spike in intensity during a slew
        indices_to_remove = np.arange(402, 436)  # 1-based 403-436 -> 0-based 402-435
        post_zwfs_images = np.delete(post_zwfs_images, indices_to_remove, axis=2)

        # The data is in normalized intensity, so we need to get the maximum value
        # of the offax PSF
        peak_offax_psf_value = coronagraph.offax.reshaped_psfs.max()
        post_zwfs_images *= np.float64(peak_offax_psf_value)

        # Create a soft circular mask for the dark hole
        y, x = np.ogrid[: speckle_shape[0], : speckle_shape[1]]
        center_y, center_x = speckle_shape[0] // 2, speckle_shape[1] // 2
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Clip speckles in the outer ring

        radius = 35
        transition = 10.0  # Width of the transition region in pixels

        # Create a mask that is 1 inside the radius, 0 outside, with a smooth transition
        mask = 1.0 - np.clip((dist_from_center - radius) / transition, 0, 1)

        # Apply the mask to all frames of the speckle data
        post_zwfs_images *= mask[..., np.newaxis]

        # Crop the image to the relevant region to remove empty space
        crop_radius = int(np.ceil(radius + transition))
        y_start = center_y - crop_radius
        y_end = center_y + crop_radius
        x_start = center_x - crop_radius
        x_end = center_x + crop_radius
        post_zwfs_images = post_zwfs_images[y_start:y_end, x_start:x_end, :]

        # Update speckle_shape for subsequent scaling calculations
        speckle_shape = post_zwfs_images.shape[:2]

        # Scaling the speckle map to match the coronagraph's size in lambda/D
        coro_shape = (coronagraph.header.naxis1, coronagraph.header.naxis2)

        # Determine the pixel scale that would be required to match the speckle map to the coronagraph
        required_pixscale = (
            speckle_pixscale * coronagraph.header.naxis1 / speckle_shape[0]
        )
        # Treat the required pixscale as a factor by which to scale the speckle map
        speckle_map_scaled = np.zeros((*coro_shape, post_zwfs_images.shape[2]))
        for i in range(post_zwfs_images.shape[2]):
            speckle_map_scaled[..., i] = flux_conserving_affine(
                post_zwfs_images[..., i],
                required_pixscale[0],
                speckle_pixscale[0],
                coro_shape,
            )
        speckle_map_scaled *= mult_factor
        # Split the data into two halves
        num_frames = speckle_map_scaled.shape[2]
        self.mid_point = num_frames // 2
        self.half1 = speckle_map_scaled[..., : self.mid_point]
        self.half2 = speckle_map_scaled[..., self.mid_point :]
        self.current_frame = 0

    def get_speckle_map(self):
        """Generate a speckle map.

        Returns:
            A 2D numpy array representing the speckle map.
        """
        idx1 = (self.current_frame + 1) % self.half1.shape[2]
        frame1 = self.half1[..., idx1]
        frame2 = self.half2[..., idx1]
        if self.do_subtraction:
            diff = median_filter(np.abs(frame2 - frame1), size=5)
        else:
            diff = median_filter(frame1, size=5)
        self.current_frame += 1
        return diff
